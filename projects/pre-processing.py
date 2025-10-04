# import module
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce
from sklearn.impute import KNNImputer

# assign dataset names
list_of_names = ['donor', 'exp_array', 'simple_somatic_mutation.open', 'specimen']

# create empty list
dataframes_list = []

# append datasets into the list
for i in range(len(list_of_names)):
	temp_df = pd.read_csv("C:/python39/"+list_of_names[i]+".tsv", sep='\t')
	dataframes_list.append(temp_df)
print(dataframes_list)


#loading dataset1

df = pd.read_csv('donor1_encoded.csv' , encoding_errors= 'replace')

df.dropna(how ='all',  axis=1, inplace=True)
df.to_csv('complied_donor.csv', index=False)

# instantiate labelencoder object
le = LabelEncoder()
#columns needs to be encoded
categorical_cols = df[["donor_sex", "donor_vital_status", "disease_status_last_followup", "donor_relapse_type"]] 
categorical_cols =categorical_cols.reset_index(drop=True)
#applying one-hot encoding
ce_OHE = ce.OneHotEncoder(df[["donor_sex", "donor_vital_status", "disease_status_last_followup", "donor_relapse_type"]])
array_hot_encoded = ce_OHE.fit_transform(categorical_cols)

#saving the encoded colums to memeory
#array_hot_encoded.to_csv('encoded_donor.csv', index=False)

#Concatenate the two dataframes : 
data_out = pd.concat([array_hot_encoded, df], axis=1)
data_out.to_csv('donor1_encoded.csv', index=False)

#loading dataset2 
df1 = pd.read_csv('complied_mutation.csv' , encoding_errors= 'replace')

#df1.dropna(how ='all',  axis=1, inplace=True)

#df1.to_csv('complied_mutation.csv', index=False)

# instantiate labelencoder object
le = LabelEncoder()
#Removing the necessary columns

#columns needs to be encoded
categorical_cols = df1[["mutation_type", "reference_genome_allele", "mutated_from_allele", "mutated_to_allele", "verification_status", "consequence_type", "aa_mutation", "cds_mutation", "gene_affected", "transcript_affected"]] 
categorical_cols =categorical_cols.reset_index(drop=True)

#applying one-hot encoding
ce_OHE = ce.OneHotEncoder(df1[["mutation_type", "reference_genome_allele", "mutated_from_allele", "mutated_to_allele", "verification_status", "consequence_type", "aa_mutation", "cds_mutation", "gene_affected", "transcript_affected"]])

array_hot_encoded = ce_OHE.fit_transform(categorical_cols)

#saving the encoded colums to memeory
#array_hot_encoded.to_csv('encoded_mutation.csv', index=False)

#Concatenate the two dataframes : 
data_out = pd.concat([array_hot_encoded, df1], axis=1)
data_out.to_csv('encoded_mutation.csv', index=False)

#loading dataset3 
df2 = pd.read_csv('specimen.csv' , encoding_errors= 'replace')

#df2.dropna(how ='all',  axis=1, inplace=True)

#df2.to_csv('complied_specimen.csv', index=False)

# instantiate labelencoder object
le = LabelEncoder()

#columns needs to be encoded
categorical_cols = df2[["specimen_type", "specimen_donor_treatment_type", "specimen_donor_treatment_type_other", "specimen_processing", "specimen_processing_other", "specimen_storage", "specimen_storage_other", "tumour_confirmed", "specimen_available"]] 
categorical_cols =categorical_cols.reset_index(drop=True)

#applying one-hot encoding
ce_OHE = ce.OneHotEncoder(df2[["specimen_type", "specimen_donor_treatment_type", "specimen_donor_treatment_type_other", "specimen_processing", "specimen_processing_other", "specimen_storage", "specimen_storage_other", "tumour_confirmed", "specimen_available"]])

array_hot_encoded = ce_OHE.fit_transform(categorical_cols)

#saving the encoded colums to memeory
#array_hot_encoded.to_csv('encoded_mutation.csv', index=False)

#Concatenate the two dataframes : 
data_out = pd.concat([array_hot_encoded, df2], axis=1)
data_out.to_csv('data/encoded_specimen.csv', index=False)
