import numpy as np
import pandas as pd
import logging
import shap
import xgboost
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt

class DropUnaryFeatures(BaseEstimator, TransformerMixin):
	"""
	Drop unary features.
	"""

	def __init__(self, features_to_drop = None):
		self.features_to_drop = features_to_drop

	def fit(self, X, y=None):
		logging.info("**********Dropping Unary Features**********")
		for col in X.columns:
			if (X.loc[:,col].nunique == 1):
				self.features_to_drop.append(col)	
		return self

	def transform(self, X):
		if self.features_to_drop:
			for col in self.features_to_drop:
				logging.info(f"{col}")
			X = X.copy()
			X = X.drop(self.features_to_drop, axis=1)
		else:
			logging.info("No features meet the requirement for exclusion.")
		logging.info("**********End of Dropping Unary Features**********\n")
		return X

class DropLowStdFeatures(BaseEstimator, TransformerMixin):
	"""
	Drop features with low standard deviation.
	"""

	def __init__(self, std_threshold = 0.01, features_to_drop=[]):
		self.std_threshold = std_threshold
		self.features_to_drop = features_to_drop

	def fit(self, X, y =None):
		logging.info("**********Dropping Low standardDev Features**********")
		for col in X.columns:
			if (X.loc[:,col].std()) <= self.std_threshold:
				self.features_to_drop.append(col)
		return self

	def transform(self, X):
		if self.features_to_drop:
			for col in self.features_to_drop:
				logging.info(f"{col}")
			X = X.copy()
			X = X.drop(self.features_to_drop, axis=1)
		else:
			logging.info("No features meet the requiremnet for exclusion.")
		logging.info("**********End of Dropping Low standardDev features**********\n")
		return X

class HighMissingFeatures(BaseEstimator, TransformerMixin):
	"""
	Drop features that have a high frequency of nan.
	"""

	def __init__(self, miss_threshold = 0.8, features_to_drop=[]):
		self.miss_threshold = miss_threshold
		self.features_to_drop = features_to_drop

	def fit(self, X, y=None):
		logging.info("**********Dropping Highly Missing Features**********")
		tot_recs = X.shape[0]
		self.miss_rate_dict = {}
		for col in X.columns:
			col_miss_rate = X.loc[:,col].isna().sum()
			col_miss_rate = col_miss_rate/tot_recs
			self.miss_rate_dict[col] = col_miss_rate
			if col_miss_rate >= self.miss_threshold:
				self.features_to_drop.append(col)
		return self

	def transform(self, X):
		if self.features_to_drop:
			for col in self.features_to_drop:
				rate = np.round(self.miss_rate_dict[col],4)*100
				logging.info(f"{col} has {rate}% nulls.")
			X = X.copy()
			X = X.drop(self.features_to_drop, axis=1)
		else:
			logging.info("No features meet the requiremnet for exclusion.")
		logging.info("**********End of Dropping Highly Missing Features**********\n")
		return X

class AddMissFlags(BaseEstimator, TransformerMixin):
	"""
	Adding missing flag indicators 
	"""

	def __init__(self, miss_threshold = 0.05, features_with_miss_flags=[]):
		self.features_with_miss_flags = features_with_miss_flags
		self.miss_threshold = miss_threshold

	def fit(self, X, y=None):
		logging.info("**********Adding Missing Flags for Features**********")
		self.miss_rate_dict = {}
		for col in X.columns:
			col_miss_rate = X.loc[:,col].isna().sum()
			col_miss_rate = col_miss_rate/X.shape[0]
			if col_miss_rate >= self.miss_threshold:
				self.features_with_miss_flags.append(col)
			self.miss_rate_dict[col] = col_miss_rate 
		return self

	def transform(self, X):
		X = X.copy()
		if self.features_with_miss_flags:
			for col in self.features_with_miss_flags:
				rate = np.round(self.miss_rate_dict[col]*100, 2)
				logging.info(f"{col} has {rate}% nulls.")
			miss_df = X[self.features_with_miss_flags].isna().astype(int).add_suffix('_miss_flag')
			logging.info(f"\n# of features Pre Missing Falgs addition {X.shape[0]}")
			X = pd.concat([X, miss_df], axis = 1)
			logging.info(f"# of features Post Missing Falgs addition {X.shape[0]}")
		else:
			logging.info("No features have missing values.")
		logging.info("**********End of Adding Missing Flags for Features**********\n")
		return X

class CorrReduction(BaseEstimator, TransformerMixin):
	"""
	Identify highly correlated features.
	Only keep the feature more correlated to the target.
	"""

	def __init__(self, threshold = 0.9, corr_method ='spearman'):
		self.threshold = threshold
		self.corr_method = corr_method

	def fit(self, X, y=None):
		logging.info("**********Removing Highly Correlated Features**********")
		temp = pd.concat([X,y], axis = 1)
		temp.columns = list(X.columns) + ['target']
		corr_matrix = temp.corr(method = self.corr_method)
		corr_matrix_ = corr_matrix.copy()
		np.fill_diagonal(corr_matrix_.values, np.nan)
		corr_matrix_ = pd.DataFrame(corr_matrix_)
		corr_matrix_.columns = temp.columns
		corr_matrix_.index =  temp.columns
		corr_matrix_.drop('target', axis = 0, inplace = True)
		target_var = corr_matrix_.loc[:, 'target']
		corr_matrix_.drop('target', axis = 1, inplace = True)
		# Only keep the upper part of the corr matrix.
		corr_matrix_ = corr_matrix_.where(np.triu(np.ones(corr_matrix_.shape)).astype(np.bool))
		corr_matrix_ = np.where(corr_matrix_ >= self.threshold, 1, 0)
		corr_matrix_ = pd.DataFrame(corr_matrix_)
		corr_matrix_.columns = X.columns
		corr_matrix_.index =  X.columns
		col_ls = list(X.columns)
		# Persist a variable exclusion dict
		self.remove_dict = {}
		remove_ls = []
		# TODO: change logic its dog shit and slow.
		for col in col_ls:
			if col not in remove_ls:
				target_corr = target_var.loc[col]
				temp_dict = {}
				temp_rm = []
				for other_col in col_ls:
					if other_col not in remove_ls:
						if (corr_matrix_.loc[col, other_col]==1):
							if target_var.loc[other_col]>=target_corr:
								remove_ls.append(col)
								self.remove_dict[col] = other_col
								temp_rm = []
								temp_dict = {}
								break
							else:
								temp_rm.append(other_col)
								temp_dict[other_col] = col
				remove_ls = remove_ls + temp_rm
				self.remove_dict.update(temp_dict)
		return self

	def transform(self, X, y = None):
		if list(self.remove_dict.keys()):
			for dropped_feat, reason_feat in self.remove_dict.items():
				logging.info(f"{dropped_feat} dropped due to high correlation with {reason_feat}")
			X = X.copy()
			X = X.drop(list(self.remove_dict.keys()), axis = 1)
		else:
			logging.info("No feature excluded due to high correlation.")
		logging.info("**********End of Removing highly correlated Features**********\n")
		return X

class TargetRateImputation(BaseEstimator, TransformerMixin):
	''' 
	Feature Imputation based on target incidence rates.

	'''

	def __init__(self, method = 'median', tol = 0.01):
		self.method = method
		self.tol = tol 
	
	# Create the imputation dictionary.
	def fit(self, X, y):
		logging.info("**********Imputing Features**********")
		temp = pd.concat([X,y], axis = 1)
		temp.columns = list(X.columns) + ['target']
		self.null_imptation_dict = {}
		self.null_percen_imputed = {}
		cols_ls = list(X.columns)
		for col in cols_ls:
			# Bucket column into 10 equal spaced bins and calc target incidence rate.
			temp['cuts'] = pd.cut(temp.loc[:,col], 10).astype('str')
			bin_means = temp.groupby('cuts').agg({'target':['mean','count']}).reset_index()
			bin_means.columns = ['cuts', 'target_rate', 'pop']
			bin_means.loc[:,'pop'] = bin_means.loc[:,'pop']/temp.shape[0]
			if 'nan' in list(bin_means['cuts']):
				null_target_rate = bin_means[bin_means['cuts']=='nan']['target_rate'].values[0]
				null_percen = bin_means[bin_means['cuts']=='nan']['pop'].values[0]
				bin_means = bin_means[bin_means['cuts']!='nan']
				bin_means = bin_means[bin_means['pop']>=self.tol]
				closest = min(bin_means.loc[:,'target_rate'].values, key = lambda k: abs(k-null_target_rate))
				cut = bin_means.loc[bin_means['target_rate'] == closest, 'cuts'].values[0]
				if self.method == 'mean':
					impute_val = np.mean(temp.loc[temp['cuts']==cut, col])
				if self.method == 'median':
					impute_val = np.median(temp.loc[temp['cuts']==cut, col])

				self.null_percen_imputed[col] = null_percen
				self.null_imptation_dict[col] = impute_val
				logging.info(f"{np.round(null_percen,3)} % of {col} is null and is imputed to {impute_val}")

		return self

	def transform(self, X, y = None):
		X = X.copy()
		for col, impute_val in self.null_imptation_dict.items():
			X.loc[:,col] = X.loc[:,col].fillna(impute_val)
		logging.info("**********End of Feature Imputation**********\n")
		return X

class ModelShapReduction(BaseEstimator, TransformerMixin):
	"""
	Build a quick model use shap to keep top variables only.
	"""

	def __init__(self, keep_top_features = 300, out_file = None):
		self.keep_top_features = keep_top_features
		self.out_file = out_file

	def fit(self, X, y=None):
		logging.info("**********Model-Shap Based Feature Reduction**********")
		set_scale_pos_wts = (sum(y==1)/sum(y==0))
		xgb_params = {'max_depth': 3
		, 'colsample_bytree': 0.8
		, 'objective': 'binary:logistic'
		, 'scale_pos_weight': set_scale_pos_wts
		, 'random_state': 42}
		DTrain = xgboost.DMatrix(X, y)
		bst_rounds = xgboost.cv(xgb_params
			, DTrain
			, num_boost_round = 1500
			, nfold = 5
			, stratified = True
			, metrics = 'auc'
			, early_stopping_rounds = 50
			, seed = 42
			, verbose_eval = False)
		best_num_booster = bst_rounds.index.max()
		del bst_rounds
		xgb_ = xgboost.train(xgb_params
			, DTrain
			, num_boost_round = best_num_booster)
		model_bytearray = xgb_.save_raw()[4:]
		def tempfunc(self=None):
			return model_bytearray
		xgb_.save_raw = tempfunc
		explainer = shap.TreeExplainer(xgb_)
		shap_values = explainer.shap_values(DTrain)
		del DTrain
		fig = shap.summary_plot(shap_values, features = X, feature_names = X.columns, show=False)
		fig = plt.gcf()
		fig.set_figheight(10.5)
		fig.set_figwidth(18.5)
		plt.savefig("top_features_shap.png", dpi=1200, bbox_inches='tight')
		shap_sum = np.abs(shap_values).mean(axis = 0)
		self.feat_importance = pd.DataFrame([X.columns.tolist(), shap_sum.tolist()]).T
		self.feat_importance.columns = ['column_names','shap_imp']
		self.feat_importance = self.feat_importance.sort_values('shap_imp', ascending = False)
		for col in list(self.feat_importance[self.feat_importance['shap_imp']==0]['column_names']):
			logging.info(f"According to shap {col} does not add prediction value.")
		self.feat_importance = self.feat_importance[self.feat_importance['shap_imp']>0]
		return self

	def transform(self, X):
		X = X.copy()
		reduced_feat_list = list(self.feat_importance.loc[:,'column_names'])
		X = X.loc[:,reduced_feat_list[0:self.keep_top_features]]
		logging.info("**********End Model-Shap Based Feature Reduction**********\n")
		return X

class RecursiveModelFeatureSelection(BaseEstimator, TransformerMixin):
	"""
	Recursively build models on the top features.
	"""

	def __init__(self, features_importance_list = None):
		self.features_importance_list = features_importance_list

	def fit(self, X, y=None):
		logging.info("\n**********Recursive Modeling for Feature Selection**********")
		self.auc_num_feats_val = {}
		auc_num_feats_train = {}
		top_feats = 1
		feat_ls = self.features_importance_list[0:top_feats]
		self.auc_num_feats_val['auc'] = {}
		self.auc_num_feats_val['auc_std'] = {}
		while len(feat_ls)<=250:
			feat_ls = self.features_importance_list[0:top_feats]
			logging.info(f"Model being built with top {len(feat_ls)} features.")
			set_scale_pos_wts = (sum(y==1)/sum(y==0))
			xgb_params = {'max_depth': 3
			, 'colsample_bytree': 0.8
			, 'objective': 'binary:logistic'
			, 'scale_pos_weight': set_scale_pos_wts
			, 'random_state': 42}
			DTrain = xgboost.DMatrix(X.loc[:,feat_ls], y)
			bst_rounds = xgboost.cv(xgb_params
				, DTrain
				, num_boost_round = 1500
				, nfold = 5
				, stratified = True
				, metrics = 'auc'
				, early_stopping_rounds = 50
				, seed = 42
				, verbose_eval = False
				)
			del DTrain
			self.auc_num_feats_val['auc'][len(feat_ls)] = bst_rounds.tail(1).iloc[0]['test-auc-mean']
			self.auc_num_feats_val['auc_std'][len(feat_ls)] = bst_rounds.tail(1).iloc[0]['test-auc-std']
			auc_num_feats_train[len(feat_ls)] = bst_rounds.tail(1).iloc[0]['train-auc-mean']
			if len(feat_ls)<=10:
				top_feats = top_feats + 1
			elif len(feat_ls)<=50:
				top_feats = top_feats + 5
			else:
				top_feats = top_feats + 20
			del bst_rounds
			if top_feats >= len(self.features_importance_list):
				break
		self.auc_num_feats_val['std_low'] = {}
		self.auc_num_feats_val['std_high'] = {}
		self.auc_num_feats_val['decay'] = {}
		for num_feat in self.auc_num_feats_val['auc'].keys():
			self.auc_num_feats_val['std_low'][num_feat] = self.auc_num_feats_val['auc'][num_feat] - 2*self.auc_num_feats_val['auc_std'][num_feat]
			self.auc_num_feats_val['std_high'][num_feat] = self.auc_num_feats_val['auc'][num_feat] + 2*self.auc_num_feats_val['auc_std'][num_feat]
			self.auc_num_feats_val['decay'][num_feat] = (100*(auc_num_feats_train[num_feat]-self.auc_num_feats_val['auc'][num_feat]))/auc_num_feats_train[num_feat]
		fig, ax1 = plt.subplots()
		fig.set_size_inches(18.5,10.5, forward = True)
		ax2 = ax1.twinx()
		ax1.plot(list(self.auc_num_feats_val['auc'].keys()), list(self.auc_num_feats_val['auc'].values()), 'b-')
		ax1.fill_between(list(self.auc_num_feats_val['std_low'].keys()), list(self.auc_num_feats_val['std_low'].values())
			, list(self.auc_num_feats_val['std_high'].values()), alpha = 0.2)
		ax2.plot(list(self.auc_num_feats_val['decay'].keys()), list(self.auc_num_feats_val['decay'].values()), 'r-')
		ax1.set_xlabel("# Top Features")
		ax1.set_ylabel("AUC-ROC", color = "b", fontsize = "large")
		ax2.set_ylabel("Decay", color = "r", fontsize = "large")
		plt.title("Perf by Top # Features")
		plt.grid()
		plt.savefig("PerfByNumFeatures.png", dpi = 1200, bbox_inches = 'tight')
		logging.info("**********End Recursive Modeling**********\n")
		return self

	def transform(X, y=None):
		return X