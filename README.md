# SCIF2001_project
SCIF2001 project code. 

### EEG_data 
Where data is stored. That is where raw data, pre processed data and features are stored. 

### Notebooks 
Contains two main notebooks 
- feature extraction - This will take the pre processed data (That is the data with the noise etc removed) and extract the features that you want to work with and save them back to the EEG data file 
- pre processing - This will take raw data, split into 30 second increments with labels and get rid of major noise in data. 

### src 
Where main code is stored - that is the main classifier code is written. 

