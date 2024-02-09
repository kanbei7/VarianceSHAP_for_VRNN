#python dataset_full.py
#python generate_dataset.py --normalizer log --featureset f20.txt
#python generate_dataset.py --normalizer minmax --featureset f20.txt
#python generate_dataset.py --normalizer mixed --featureset f20.txt
#python generate_dataset.py --normalizer none --featureset f20.txt
#python generate_dataset.py --normalizer zscore --featureset f20.txt

#python generate_dataset.py --normalizer log --featureset f20.txt --binary_mask binary
#python generate_dataset.py --normalizer minmax --featureset f20.txt --binary_mask binary
#python generate_dataset.py --normalizer mixed --featureset f20.txt --binary_mask binary
#python generate_dataset.py --normalizer none --featureset f20.txt --binary_mask binary
#python generate_dataset.py --normalizer zscore --featureset f20.txt --binary_mask binary

#python generate_dataset.py --normalizer log
#python generate_dataset.py --normalizer minmax
#python generate_dataset.py --normalizer mixed
python generate_dataset.py --normalizer none --binary_mask binary
python generate_dataset.py --normalizer zscore --binary_mask binary
python check_lengths.py