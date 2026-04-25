#!/bin/bash
if [ $# -eq 0 ]; then
        echo "You need to give a lab name: './setLabName.sh myLabName'"
else
    mv imitation_lab.py $1.py
    mv imitation_lab.md $1.md
    sed -i 's/Emio Imitation Learning/'$1'/g' $1.md
    sed -i 's/"name": "imitation lab"/"name": "'$1'"/g' lab.json
    sed -i 's/"filename": "imitation_lab.md"/"filename": "'$1'.md"/g' lab.json
    sed -i 's/"title": "Imitation Lab"/"title": "'$1'"/g' lab.json
    echo "Done renaming lab: '$1'"
fi
