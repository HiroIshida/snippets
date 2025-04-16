if [ ! -d .ssh ]
then
    echo ".ssh directory not found."
    ln -sf $HOME/.ssh .
fi
tar -chz . |docker build -t myhumble -

