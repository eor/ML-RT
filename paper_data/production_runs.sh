#!/bin/sh

# GitHub has a 100MB filesize limit at this time, so please download the data using this script.

BASEURL='https://drive.google.com/uc?id='

FILE1='1MNZy1J5n-cfk40H-wnO7PIrtVO-wMw2i'
FILE1NAME='production_runs.zip'
CHECKSUM1='0b1370281ff12785b61c70d172bea3f6'

dl() {

        gdown $BASEURL$FILE1
}
  

check() {

        VAR1=$(md5sum ${FILE1NAME} | awk '{print $1}')

        # 1
        if [ "$CHECKSUM1" = "$VAR1" ]; then

            echo "Checksum OK: "$FILE1NAME
            
        else
        
            echo "Checksum NOT ok: "$FILE1NAME       
            
        fi
        
}

extract(){
        unzip $FILE1NAME
}


if ! (command -v gdown &> /dev/null)
then
    echo "Error: gdown could not be found. Exiting"
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

if ! (command -v unzip &> /dev/null)
then
    echo "Error: unzip could not be found. Exiting"
    exit

else
    extract
fi


