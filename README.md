This project performs unsupervised learning in order to create dataset of lego images 
 
 How to setup and run project

0. Install requirements - "requirements.txt"
1. Create "feature_extraction" directory and unzip features_extracted.zip there and unpack folder
2. Copy "main_datasate_directory\dataset_augmented" directory with augmented data from UnsupervisedTransferLearning
3. Create directories: "clustering_1", "clustering_2", "clustering_with_best_k", "result" for results
4. Run clustering_stage_1.py, it performs initial clustering and outputs to clustering_1
5. Run clustering_stage_2.py, it conducts clustering for all possible k and saves metrics to clustering_2
6. Run clustering_with_best_k.py, it uses one of metric (elbow, silhoutte score) to find best k, runs KMeans with that k and outputs to clustering_with_best_k
7. Run restore.py, it restores folders with imatges based on output from the previous step

###

visualize.py - basic charts generation for output data from clustering_stage_2
read_features.py - utils to read features vector
