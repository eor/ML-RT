#!/bin/sh

# GitHub has a 100MB filesize limit at this time, so please download the data using this script.

BASEURL="https://www.astro.rug.nl/~krause/static/ML-RT/053_data_set/"

FILE1="data_Hprofiles.npy"
FILE2="data_Tprofiles.npy"
FILE3="data_parameters.npy"

CHECKSUM1="62375d3a72ee14f12e1d96a589d3c1de"
CHECKSUM2="bfbfbb279b41aed19c39d933633a67e3"
CHECKSUM3="a8e1de335d896a7fcd573d401836a23e"



dl() {

        wget -c $BASEURL$FILE1
        wget -c $BASEURL$FILE2
        wget -c $BASEURL$FILE3
}


check() {

        VAR1=$(md5sum ${FILE1} | awk '{print $1}')
        VAR2=$(md5sum ${FILE2} | awk '{print $1}')
        VAR3=$(md5sum ${FILE3} | awk '{print $1}')
        
        # 1
        if [ "$CHECKSUM1" = "$VAR1" ]; then

            echo "Checksum OK: "$FILE1
            
        else
        
            echo "Checksum NOT ok: "$FILE1       
            
        fi
        
        # 2
        if [ "$CHECKSUM2" = "$VAR2" ]; then

            echo "Checksum OK: "$FILE2
            
        else
            echo "Checksum NOT ok: "$FILE2
        fi
        
        # 3
        if [ "$CHECKSUM3" = "$VAR3" ]; then

            echo "Checksum OK: "$FILE3
            
        else
            echo "Checksum NOT ok: "$FILE3
        fi        
        
}


if ! (command -v wget &> /dev/null)
then
    echo "Error: wget could not be found. Exiting"
    exit

else
    dl
fi


if ! (command -v md5sum &> /dev/null)
then
    echo "Warning: md5sum could not be found. Can't verify file integrity"
    exit
    
else
    check
fi










