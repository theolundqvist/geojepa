#if [ $(id -u) -ne 0 ]
#  then echo Please run this script as root or using sudo!
#  exit
#fi

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/theo/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/theo/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/theo/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/theo/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<


conda activate ag || { echo "Error: conda env 'ag' does not exist, make sure to run script with \"sudo -E\" to pass environment"; exit 1; }
conda activate geojepa || { echo "Error: conda env 'geojepa' does not exist, make sure to run script with \"sudo -E\" to pass environment."; exit 1; }
