###Instructions to use the SaDe Codes.

####Structure of the Code:
1. The main learners are present in the /src folder, along with the MaxL optimizer.
2. There is a separate folder for each use case, Experiments.py is the main experiments file in each such folder.


####Running the Experiments
1. Each use case folder has a run_command.txt file that contains the command to run the Experiments.py through terminal.
2. Run commands have command line arguments that will be used in the experiments. 
3. Each run command has the following structure:
    
    python -m scoop -vv -n number_of_cores_to_use Experiments.py '[error/classification thresholds]' '1 if knowledge constraint is enforced' '[number_of_epochs]' '[learning range values]' '[batch_sizes]' 'numeric value of the experiment number'
   
4. Experiment number ranges from 1 to 10 for the loan and expense use case and from 21 to 30 for the music use case, where the random seed is set as experiment_number^3    
5. Each use case folder contains a Supplement.py file that contains functions that are used for experiments.
6. Each use case folder contains a Regularization based baseline script. (e.g. MusicGenreClassificationRegularization.py)
7. Each use case folder contains a SaDe learner (e.g. LinearClassifierMusic.py) specific to that use case that subsumes the main learners from /src (path to the /src folders must be added to these learner files).
8. Each use case folder contains a cross validation script (e.g. CrossValidationMultiClass.py) for SaDe models. 
10. Datasets for loan and expense use cases are present in the respective folders. Dataset for the music use case is not provided because it comes from a proprietary source.
11. In addition, there is a folder called 'MusicGenreClassificationFuMalik' that constraints the scripts to learn models purely with Fu-Malik. These results are added in the technical appendix.

####Dependencies

1. For each experiment, z3 solver (version 4.8.10) is used.
2. For all the other packages, the latest version are used.