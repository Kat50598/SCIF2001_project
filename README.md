# SCIF2001_project
SCIF2001 project code. 

### EEG_data 
Where data is stored. That is where raw data, pre processed data and features are stored. 

### Notebooks 
Contains two main notebooks 
- feature extraction - This will take the pre processed data (That is the data with the noise etc removed) and extract the features that you want to work with and save them back to the EEG data file 
- pre processing - This will take raw data, split into 30 second increments with labels and get rid of major noise in data. 

### src 
#### Data_pre_processing 
After you have put the downloaded files from the paper into EEG if you rename the downloaded file from the paper to specifically be ds005207 then run src it should automatically iterate through and clean the data and put them into the new files called sub-xxx_cleaned-epo.fif. 

The file takes only PSG data from each subject and of that only the EEG data, then chops it up into 30 second increments, labels it and puts it into a table. If you want to see how visualise that final data try running data_clean_visualise and it will take the 



