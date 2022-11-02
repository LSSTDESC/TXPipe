NERSC Shifter Installation
==========================

The easiest way to use TXPipe on NERSC is via the Shifter system, which creates a container with all the requirements pre-built.

To use this, first run::

    source /global/cfs/cdirs/lsst/groups/WL/users/zuntz/setup-txpipe

This will create a command ``tx``. You prefix any command with this to run it inside the container.

For example::

    tx ceci examples/metacal/pipeline.yml

This somtimes breaks if you have previously run ``conda init`` on the system, because the conda python is found even inside the container. You can avoid this by modifying your ``$HOME/.bashrc`` file to put these test lines around the conda setup commands::

    if [ "${TX_DOCKER_VERSION}" == "" ]
    then
        # >>> conda initialize >>>
        ...
        ...
        # <<< conda initialize <<<
    fi
