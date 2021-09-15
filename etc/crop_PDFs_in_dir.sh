#!/bin/bash


DIR=$1

# check if pdfcrop is available

if ! (command -v pdfcrop &> /dev/null)
then
    echo "Error: pdfcrop could not be found. Exiting"
    exit 1
fi


# check if an argument has been suplied


if [[ $# -eq 0 ]] ; then
    echo "Usage: $0 <directory path>"
    echo 'Exiting'
    exit 1
fi


# check if path exists

if [ -d "$DIR" -a ! -h "$DIR" ]
then
   echo "$DIR found. Converting PDFs (if there are any) ..."
else
   echo "Error: $DIR not found or is symlink"
   exit 1
fi


# cd to dir, do the thing

cd ${DIR}

for FILE in ./*.pdf; do
  pdfcrop  "${FILE}"  "${FILE}"
done
