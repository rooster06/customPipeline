import pandas as pd
import numpy as np
import logging
from datetime import datetime
import preprocessing as pp
import config

if __name__ == '__main__':
	
	# start logger
	log_file = 'log_model_dev{date:%Y-%m-%d_%H:%M:%S}.log'.format(date=datetime.now())
	logging.basicConfig(format='%(message)s', filename=log_file, level=logging.INFO)
	logging.info('Starting model development!\n')

	# load the dataset
	data = pd.read_csv(config.PATH_TO_BUILD_DATASET)

	# seperate out into independent features and target
	target = data.loc[:, config.TARGET]
	data = data.drop(config.TARGET, axis = 1)

	# drop unary features 
	drop_unary = pp.DropUnaryFeatures()
	drop_unary.fit(data)
	data = drop_unary.transform(data)

	# drop low std deviation features
	lowstdev = pp.DropLowStdFeatures(std_threshold = 0.02)
	lowstdev.fit(data)
	data = lowstdev.transform(data)

	# remove highly missing features 
	miss_feats = pp.HighMissingFeatures(miss_threshold = 0.18)
	miss_feats.fit(data)
	data = miss_feats.transform(data)

	# Add Miss Flags for features
	print(data.shape)
	miss_flags = pp.AddMissFlags(miss_threshold = 0.02)
	miss_flags.fit(data)
	data = miss_flags.transform(data)
	print(data.shape)

	# intermediate step - Impitation for correlation reduction
	target_based_imputation = pp.TargetRateImputation()
	target_based_imputation.fit(data, target)
	data_temp = target_based_imputation.transform(data)
	corr_reduction = pp.CorrReduction(threshold = 0.2)
	corr_reduction.fit(data_temp, target)
	data_temp = corr_reduction.transform(data_temp)
	keep_ls = data_temp.columns.tolist()
	del data_temp
	data = data.loc[:,keep_ls]
	print(data.shape)

	# model based reduction
	model_based_reduction = pp.ModelShapReduction()
	model_based_reduction.fit(data, target)
	feat_importance = model_based_reduction.feat_importance['column_names'].tolist()
	# recursive model building 
	recursive_model_feature_selection = pp.RecursiveModelFeatureSelection(features_importance_list = feat_importance)
	recursive_model_feature_selection.fit(data, target)

	# final model building 

	# evaluation
