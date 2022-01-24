array=("1A1-G09" "1P1-G02" "1P2-G09" "2A1-L08" "2A1-M14" "2P2-I03")
for name in "${array[@]}"
do
    datalist=$(find ~/Downloads/proceedings/ -name *$name*)
    for data in $datalist
    do
        cp $data ./data/
    done
done
