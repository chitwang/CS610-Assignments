blk_values=(2 4 8 16 32 64 128)

for i in "${blk_values[@]}"; do
    for cnt in {1..3}; do
        # echo "$i" "$cnt"                                
		./sol "$i" "$i" "$i"
    done
done
