import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data

from src import log
from src.utils.graph_utils import build_spanning_tree_edge, get_angle, triplets
from torch import Tensor
from torch.nn import ModuleList, Parameter
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.glob import global_add_pool
from torch_scatter import scatter_add


class GeometryEncoder(nn.Module):
    r"""
    Uses Polygon GNN from https://github.com/dyu62/PolyGNN, to encode a geometry graph into a fixed-size embedding.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = Smodel(*args, **kwargs)

    def forward(
        self, node_positions, intra_edges, inter_edges, node_to_feature
    ) -> Tensor:
        r"""
        Args:
            node_positions (Tensor): Node positions [N, 2]`.
            intra_edges (Tensor): node gid - node gid `[2, E_intra]`.
            inter_edges (Tensor): node gid - node gid `[2, E_inter]`.
            node_to_feature (Tensor): Node to OSM feature map `[N]`.
        """
        combined_edges = torch.cat([intra_edges, inter_edges], dim=1)
        num_nodes = node_positions.shape[0]
        device = node_positions.device
        if self.training:
            edge_weight = torch.rand(combined_edges.shape[1]) + 1
            undirected_spanning_edge = build_spanning_tree_edge(
                combined_edges, edge_weight, num_nodes=num_nodes
            )
            edge_set_1 = set(map(tuple, intra_edges.t().tolist()))
            edge_set_2 = set(map(tuple, undirected_spanning_edge.t().tolist()))
            common_edges = edge_set_1.intersection(edge_set_2)
            common_edges_tensor = (
                torch.tensor(list(common_edges), dtype=torch.long).t().to(device)
            )
            spanning_edge = torch.cat([inter_edges, common_edges_tensor], 1)
            combined_edges = spanning_edge

        edge_index_2rd, num_triplets_real, edx_jk, edx_ij = triplets(
            combined_edges, num_nodes
        )
        input_feature = torch.zeros([num_nodes, self.model.h_dim], device=device)
        output = self.model(
            input_feature,
            node_positions,
            edge_index_2rd,
            edx_jk,
            edx_ij,
            inter_edges.shape[1],
        )
        output = torch.cat(output, dim=1)
        graph_embeddings = global_add_pool(output, node_to_feature)
        return graph_embeddings.clamp(max=1e6)


# Modified from https://github.com/dyu62/PolyGNN
class Smodel(nn.Module):
    def __init__(
        self,
        h_dim=512,
        localdepth=1,
        num_interactions=4,
        finaldepth=4,
        share="0",
        batchnorm="True",
    ):
        super().__init__()
        self.h_dim = h_dim
        self.localdepth = localdepth
        self.num_interactions = num_interactions
        self.finaldepth = finaldepth
        self.batchnorm = batchnorm
        self.activation = nn.ReLU()
        self.att = Parameter(torch.ones(4), requires_grad=True)

        num_gaussians = (1, 1, 1)
        self.mlp_geo = ModuleList()
        for i in range(self.localdepth):
            if i == 0:
                self.mlp_geo.append(Linear(sum(num_gaussians), h_dim))
            else:
                self.mlp_geo.append(Linear(h_dim, h_dim))
            if self.batchnorm == "True":
                self.mlp_geo.append(nn.BatchNorm1d(h_dim))
            self.mlp_geo.append(self.activation)

        self.interactions = ModuleList()
        for i in range(self.num_interactions):
            block = SPNN(
                in_ch=self.h_dim,
                hidden_channels=self.h_dim,
                activation=self.activation,
                finaldepth=self.finaldepth,
                batchnorm=self.batchnorm,
                num_input_geofeature=self.h_dim,
            )
            self.interactions.append(block)
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.mlp_geo:
            if isinstance(lin, Linear):
                torch.nn.init.xavier_uniform_(lin.weight)
                lin.bias.data.fill_(0)
        for i in self.interactions:
            i.reset_parameters()

    def single_forward(
        self, input_feature, coords, edge_index_2rd, edx_jk, edx_ij, num_edge_inside
    ):
        i, j, k = edge_index_2rd
        distance_ij = (coords[j] - coords[i]).norm(p=2, dim=1)
        distance_jk = (coords[j] - coords[k]).norm(p=2, dim=1)
        theta_ijk = get_angle(coords[j] - coords[i], coords[k] - coords[j])
        geo_encoding_1st = distance_ij[:, None]
        geo_encoding = torch.cat(
            [geo_encoding_1st, distance_jk[:, None], theta_ijk[:, None]], dim=-1
        )
        for lin in self.mlp_geo:
            geo_encoding = lin(geo_encoding)
        node_feature = input_feature
        node_feature_list = []
        for interaction in self.interactions:
            node_feature = interaction(
                node_feature,
                geo_encoding,
                edge_index_2rd,
                edx_jk,
                edx_ij,
                num_edge_inside,
                self.att,
                coords.shape[0],
            )
            node_feature_list.append(node_feature)
        return node_feature_list

    def forward(
        self, input_feature, coords, edge_index_2rd, edx_jk, edx_ij, num_edge_inside
    ):
        output = self.single_forward(
            input_feature, coords, edge_index_2rd, edx_jk, edx_ij, num_edge_inside
        )
        return output


class SPNN(torch.nn.Module):
    def __init__(
        self,
        in_ch,
        hidden_channels,
        activation=torch.nn.ReLU(),
        finaldepth=3,
        batchnorm="True",
        num_input_geofeature=13,
    ):
        super().__init__()
        self.activation = activation
        self.finaldepth = finaldepth
        self.batchnorm = batchnorm
        self.num_input_geofeature = num_input_geofeature

        self.WMLP_list = ModuleList()
        for _ in range(4):
            WMLP = ModuleList()
            for i in range(self.finaldepth + 1):
                if i == 0:
                    WMLP.append(
                        Linear(
                            hidden_channels * 3 + num_input_geofeature, hidden_channels
                        )
                    )
                else:
                    WMLP.append(Linear(hidden_channels, hidden_channels))
                if self.batchnorm == "True":
                    WMLP.append(nn.BatchNorm1d(hidden_channels))
                WMLP.append(self.activation)
            self.WMLP_list.append(WMLP)
        self.reset_parameters()

    def reset_parameters(self):
        for mlp in self.WMLP_list:
            for lin in mlp:
                if isinstance(lin, Linear):
                    torch.nn.init.xavier_uniform_(lin.weight)
                    lin.bias.data.fill_(0)

    def forward(
        self,
        node_feature,
        geo_encoding,
        edge_index_2rd,
        edx_jk,
        edx_ij,
        num_edge_inside,
        att,
        num_nodes,
    ):
        i, j, k = edge_index_2rd
        if node_feature is None:
            concatenated_vector = geo_encoding
        else:
            node_attr_0st = node_feature[i]
            node_attr_1st = node_feature[j]
            node_attr_2 = node_feature[k]
            concatenated_vector = torch.cat(
                [
                    node_attr_0st,
                    node_attr_1st,
                    node_attr_2,
                    geo_encoding,
                ],
                dim=-1,
            )
        x_i = concatenated_vector

        edge1_edge1_mask = (edx_ij < num_edge_inside) & (edx_jk < num_edge_inside)
        edge1_edge2_mask = (edx_ij < num_edge_inside) & (edx_jk >= num_edge_inside)
        edge2_edge1_mask = (edx_ij >= num_edge_inside) & (edx_jk < num_edge_inside)
        edge2_edge2_mask = (edx_ij >= num_edge_inside) & (edx_jk >= num_edge_inside)
        masks = [edge1_edge1_mask, edge1_edge2_mask, edge2_edge1_mask, edge2_edge2_mask]

        x_output = torch.zeros(
            x_i.shape[0], self.WMLP_list[0][0].weight.shape[0], device=x_i.device
        )
        for index in range(4):
            WMLP = self.WMLP_list[index]
            x = x_i[masks[index]]
            for lin in WMLP:
                x = lin(x)
            x = F.leaky_relu(x) * att[index]
            x_output[masks[index]] += x

        return scatter_add(x_output, i, dim=0, dim_size=num_nodes)


def load_geometry_encoder_pretrained(
    model_path: str, device: torch.device = torch.device("cpu")
) -> GeometryEncoder:
    saved_dict = torch.load(
        model_path,
        map_location=device,
    )
    model = Smodel(
        h_dim=saved_dict["args"].h_ch,
        localdepth=saved_dict["args"].localdepth,
        num_interactions=saved_dict["args"].num_interactions,
        finaldepth=saved_dict["args"].finaldepth,
        share=saved_dict["args"].share,
        batchnorm=saved_dict["args"].batchnorm,
    )
    model.to(device)
    for key in [k for k in saved_dict["model"].keys()]:
        if "mlp_geo_backup" in key or "translinear" in key:
            del saved_dict["model"][key]
    try:
        model.load_state_dict(saved_dict["model"], strict=True)
    except OSError:
        log.info("loadfail: ", str)
    encoder = GeometryEncoder()
    encoder.model = model
    return encoder
