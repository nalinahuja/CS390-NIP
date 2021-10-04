# Developed By Nalin Ahuja, ahuja15

# Algorithm List
ALGORITHMS=("guesser" "tf_net" "tf_conv")

# Datasets List
DATASETS=("mnist_d" "mnist_f" "cifar_10" "cifar_100_f" "cifar_100_c")

# End Program Constants--------------------------------------------------------------------------------------------------------------------------------------------------

# Get Terminal Columns
columns=$(command tput cols)

# Separator Format
sfmt=$(command printf "%-${columns}s")

# Iterate Over Algorithms
for algorithm in ${ALGORITHMS[@]}; do
  # Iterate Over Datasets
  for dataset in ${DATASETS[@]}; do
    # Run Program With Arguments
    command python3 ./lab2.py ${algorithm} ${dataset}

    # Print Separator
    command echo -e "\n${sfmt// /-}\n"
  done
done

# End File---------------------------------------------------------------------------------------------------------------------------------------------------------------
