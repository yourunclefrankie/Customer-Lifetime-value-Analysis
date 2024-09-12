#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from lifetimes import BetaGeoFitter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# In[2]:


Tranx_df = pd.read_csv("Online_Retail.csv",encoding='ISO-8859-1')


# In[3]:


Tranx_df


# In[4]:


Tranx_df.duplicated().sum()


# In[5]:


Tranx_df.drop_duplicates()


# In[6]:


missing_values = Tranx_df.isnull().sum()


# In[7]:


plt.figure(figsize=(8, 6))
missing_values.plot(kind='bar', color='skyblue')
plt.title('Missing Values in Each Column')
plt.xlabel('Columns')
plt.ylabel('Number of Missing Values')
plt.xticks(rotation=0)
plt.show()


# In[8]:


Tranx_df.dropna(subset=['CustomerID'], inplace=True)


# In[9]:


Tranx_df.isnull().sum()


# In[10]:


Tranx_df


# ### Dataframe Details 

# In[11]:


Tranx_df.info()


# #### convert Invoice Date columnn to Date 

# In[12]:


Tranx_df['InvoiceDate'] = pd.to_datetime(Tranx_df['InvoiceDate'])


# In[13]:


Tranx_df_max =  Tranx_df['InvoiceDate'].max()


# In[14]:


Tranx_df_min =  Tranx_df['InvoiceDate'].min()


# #### Determine a cutoff date for 90 days 

# In[15]:


Tranx_df_min


# In[16]:


Tranx_df_max


# In[17]:


n_days= 90 

cut_off = Tranx_df_max - pd.to_timedelta(n_days, unit='days')


# In[18]:


print(cut_off)


# In[19]:


(cut_off - Tranx_df_min).days


# In[20]:


temporal_in = Tranx_df[Tranx_df['InvoiceDate'] <= cut_off]
temporal_out = Tranx_df[Tranx_df['InvoiceDate'] > cut_off]


# In[21]:


temporal_in


# In[22]:


Input_df = temporal_in.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (cut_off - x.max()).days, 
    'InvoiceNo': 'count',  
    'UnitPrice': 'sum'  
}).rename(columns={
    'InvoiceDate': 'recency',
    'InvoiceNo': 'frequency',
    'UnitPrice': 'Monetary'
}).assign(T=lambda x: (cut_off - temporal_in.groupby('CustomerID')['InvoiceDate'].min()).dt.days)


# In[23]:


Input_df


# In[24]:


Input_df['AVG_PURCHASE'] = Input_df['Monetary']/Input_df['frequency']


# In[25]:


clv_df = temporal_out.groupby('CustomerID').agg({
    'UnitPrice': 'sum'
}).rename(columns={'UnitPrice': 'CLV'})

print("CLV:\n", clv_df.head())


# In[26]:


clv_df


# In[27]:


final_df = Input_df.merge(clv_df, on='CustomerID', how='left').fillna(0)


# In[28]:


final_df


# In[29]:


final_df.columns.tolist()


# In[30]:


plt.figure(figsize=(12, 6))
sns.boxplot(data=final_df)
plt.show()


# In[31]:


features = ['recency', 'frequency', 'Monetary', 'T', 'AVG_PURCHASE', 'CLV']

plt.figure(figsize=(10,4))
for i, feature in enumerate(features):
    plt.subplot(2, 3, i + 1)
    sns.histplot(final_df[feature], bins=30, kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()


# In[32]:


final_df.info()


# In[33]:


def cap_outliers(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return np.where(column < lower_bound, lower_bound,
                    np.where(column > upper_bound, upper_bound, column))


# In[34]:


final_df = final_df.apply(cap_outliers, axis=0)


# In[35]:


final_df


# In[36]:


plt.figure(figsize=(12, 6))
sns.boxplot(data=final_df)
plt.show()


# In[37]:


features = ['recency', 'frequency', 'Monetary', 'T', 'AVG_PURCHASE', 'CLV']

plt.figure(figsize=(10,4))
for i, feature in enumerate(features):
    plt.subplot(2, 3, i + 1)
    sns.histplot(final_df[feature], bins=30, kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()


# In[38]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

features = ['recency', 'frequency', 'Monetary']
df_log_transformed = final_df.copy()
df_log_transformed['Monetary'] = np.log1p(df_log_transformed['Monetary'])
df_log_transformed['recency'] = np.log1p(df_log_transformed['recency'])
df_log_transformed['T'] = np.log1p(df_log_transformed['T'])
df_log_transformed['AVG_PURCHASE'] = np.log1p(df_log_transformed['AVG_PURCHASE'])

plt.figure(figsize=(16, 8))
for i, feature in enumerate(features):
    plt.subplot(2, 3, i+1)
    sns.histplot(y=df_log_transformed[feature], color='skyblue')
    plt.title(f'Boxplot of {feature} (Log-Transformed)' if feature in df_log_transformed.columns else f'Boxplot of {feature}')
    plt.ylabel(feature)

plt.tight_layout()
plt.show()


# In[39]:


corr= final_df.corr()


# In[40]:


plt.figure(figsize=(8, 5))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()


# In[41]:


Y = final_df['CLV']


# In[42]:


X = final_df.drop(columns =['CLV'])


# In[43]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[44]:


lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)


# In[45]:


gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)


# In[46]:


def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R-squared: {r2:.2f}")
    

print("Random Forest:")
evaluate_model(rf_model, X_test, y_test)


print("Linear Regression:")
evaluate_model(lr_model, X_test, y_test)


print("\nGradient Boosting:")
evaluate_model(gb_model, X_test, y_test)


# In[47]:


y_pred_gb =gb_model.predict(X_test)


# In[48]:


y_pred_lr = lr_model.predict(X_test)


# In[49]:


y_pred_rf = rf_model.predict(X_test)


# In[50]:


y_gb = gb_model.predict(X)


# In[51]:


final_df['predict'] = y_gb


# In[52]:


final_df


# In[53]:


final_df


# In[54]:


gb_model = GradientBoostingRegressor(random_state=42)

param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=gb_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)

grid_search.fit(X_train, y_train)

print(f"Best hyperparameters: {grid_search.best_params_}")
best_gb_model = grid_search.best_estimator_


# In[55]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


y_pred = best_gb_model.predict(X_test)


rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R-squared: {r2:.2f}")


# In[56]:


snapshot_date = Tranx_df['InvoiceDate'].max() + pd.Timedelta(days=1)
customer_metrics_train =Tranx_df.groupby('CustomerID').agg(
     recency=('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
    frequency=('InvoiceDate', 'count'),
     monetary=('UnitPrice', 'mean'),
           T=('InvoiceDate', lambda x: (snapshot_date - x.min()).days)
).reset_index()
  

snapshot_date_holdout = Tranx_df['InvoiceDate'].max() + pd.Timedelta(days=1)
customer_metrics_holdout = Tranx_df.groupby('CustomerID').agg(
    actual_clv=('UnitPrice', 'sum')
).reset_index()

bgf = BetaGeoFitter(penalizer_coef=0.0)
bgf.fit(customer_metrics_train['frequency'], customer_metrics_train['recency'], customer_metrics_train['T'])

prediction_horizon = 90
customer_metrics_train['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(
    prediction_horizon, customer_metrics_train['frequency'], customer_metrics_train['recency'], customer_metrics_train['T']
)
customer_metrics_train['predicted_clv_pareto'] = customer_metrics_train['predicted_purchases'] * customer_metrics_train['monetary']


results = pd.merge(customer_metrics_train[['CustomerID', 'predicted_clv_pareto']],
                   customer_metrics_holdout[['CustomerID', 'actual_clv']],
                   on='CustomerID', how='left').fillna(0)

mse = mean_squared_error(results['actual_clv'], results['predicted_clv_pareto'])
rmse = np.sqrt(mse)
r_squared = r2_score(results['actual_clv'], results['predicted_clv_pareto'])

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R-squared: {r_squared}")



# In[57]:


compare = pd.merge(results,final_df ,on = ['CustomerID'] ).drop(columns=['actual_clv','recency','frequency','AVG_PURCHASE'])


# In[58]:


compare


# In[59]:


compare_subset = compare.head(8)

plt.figure(figsize=(20, 8))

sns.lineplot(data=compare_subset, x="predicted_clv_pareto", y="CustomerID", label='Predicted CLV Pareto', marker='o')

sns.lineplot(data=compare_subset, x="predict", y="CustomerID", label='Predicted CLV Gradient Boost', marker='o')

sns.lineplot(data=compare_subset, x="CLV", y="CustomerID", label='Actual CLV', marker='o')

plt.title('Comparison of CLV Predictions and Actual CLV for the First 20 CustomerIDs')
plt.xlabel('CLV Value')
plt.ylabel('CustomerID')
plt.legend()


# In[60]:


y_test_df = pd.DataFrame({'y_pred_gb': y_pred_gb,
                           'y_pred_lr': y_pred_lr,
                           'y_pred_rf': y_pred_rf})


# In[61]:


y_test_df["customerID"] = compare['CustomerID']


# In[62]:


y_test_df["CLV"] = compare['CLV']


# In[63]:


y_test_df


# In[64]:


y_test_df_ = y_test_df.head(8)

plt.figure(figsize=(20, 8))

sns.lineplot(data= y_test_df_, x="y_pred_gb", y="customerID", label='Predicted gradient Boost', marker='o')

sns.lineplot(data= y_test_df_,x="y_pred_lr", y="customerID", label='predicted linear Regression', marker='o')

sns.lineplot(data= y_test_df_, x="y_pred_rf", y="customerID", label='Predicted random forest', marker='o')

sns.lineplot(data= y_test_df_, x="CLV", y="customerID", label='Actual CLV ', marker='o')

plt.title('Comparison of CLV Predictions and Actual CLV for the First 20 CustomerIDs')
plt.xlabel('CLV Value')
plt.ylabel('CustomerID')
plt.legend()


# In[65]:


final_df


# In[66]:


final_df= final_df.drop(columns=['predict'])


#  ###  Segment Dataset 

# In[67]:


scaler = StandardScaler()
final_df_sc = scaler.fit_transform(final_df)


# ###  determine the number of components 

# In[68]:


pca = PCA()
pca.fit(final_df_sc)

explained_variance = pca.explained_variance_ratio_

cumulative_explained_variance = np.cumsum(explained_variance)


plt.figure(figsize=(6,3))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', label='Individual Explained Variance')
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--', label='Cumulative Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Explained Variance')
plt.legend()
plt.grid(True)
plt.show()


# In[69]:


pca = PCA(n_components= 4)
PC_final_df = pca.fit_transform(final_df_sc)


# In[70]:


wcss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(PC_final_df)
    wcss.append(kmeans.inertia_)


# In[71]:


plt.plot(range(1, 10), wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.show()


# In[72]:


kmeans =   KMeans(n_clusters=3, init="random",random_state=0)

Y_kmeans = kmeans.fit_predict(PC_final_df)


# In[73]:


Y_kmeans


# In[74]:


sum(pca.explained_variance_ratio_)


# In[75]:


final_df["cluster"] = Y_kmeans


# In[76]:


final_df


# In[77]:


baseline_labels = kmeans.labels_

def permutation_importance_kmeans(model, X, baseline_labels, n_repeats=30, random_state=0):
    rng = np.random.RandomState(random_state)
    importance_scores = np.zeros(X.shape[1])
    
    for col in range(X.shape[1]):
        scores = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            rng.shuffle(X_permuted[:, col])
            model.fit(X_permuted)
            permuted_labels = model.labels_
            score = adjusted_rand_score(baseline_labels, permuted_labels)
            scores.append(score)
        importance_scores[col] = np.mean(scores)
    
    return 1 - importance_scores  

perm_importance = permutation_importance_kmeans(kmeans,PC_final_df, baseline_labels)

feature_importance_df = pd.DataFrame({
    'Feature': ['PC1', 'PC2','PC3','PC4'],
    'Importance': perm_importance
})

print("\nFeature importance based on permutation importance:")
print(feature_importance_df)


# In[78]:


pca_components = pd.DataFrame(pca.components_, columns=final_df.drop(columns=['cluster']).columns, index=['PC1','PC2','PC3','PC4'])
print("\nContribution of original features to each principal component:")
print(pca_components)


# In[79]:


pca_df = pd.DataFrame(PC_final_df, columns=['PC1', 'PC2','PC3','PC4'])
pca_df['cluster'] = kmeans.labels_
cluster_summary = pca_df.groupby('cluster').mean()
print("\nMean values of each principal component in each cluster:")
print(cluster_summary)


# In[80]:


for pc in ['PC1', 'PC2','PC3','PC4']:
    print(f"\n{pc} is dominated by:")
    sorted_features = pca_components.loc[pc].abs().sort_values(ascending=False)
    print(sorted_features)


# #### ------------------------- FIRST CLUSTER ----------------------------------------------

# In[81]:


final_cluster_0 = final_df[final_df['cluster']==0]


# In[82]:


YC = final_cluster_0['CLV']


# In[83]:


YC


# In[84]:


XC = final_cluster_0.drop(columns=['CLV','cluster'])


# In[85]:


XC


# In[86]:


XC_train, XC_test, yC_train, yC_test = train_test_split(XC, YC, test_size=0.2, random_state=42)


# In[87]:


lr_model = LinearRegression()
lr_model.fit(XC_train, yC_train)

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(XC_train, yC_train)


gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(XC_train, yC_train)


# In[88]:


def evaluate_model(model, XC_test, yC_test):
    yC_pred = model.predict(XC_test)
    rmse = mean_squared_error(yC_test, yC_pred, squared=False)
    mae = mean_absolute_error(yC_test, yC_pred)
    r2 = r2_score(yC_test, yC_pred)
    print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R-squared: {r2:.2f}")
    


print("\nGradient Boosting:")
evaluate_model(gb_model, XC_test, yC_test)


# In[89]:


print("Random Forest:")
evaluate_model(rf_model, XC_test, yC_test)


print("Linear Regression:")
evaluate_model(lr_model, XC_test, yC_test)


# In[90]:


y_test_df = pd.DataFrame({'y_pred_gb': y_pred_gb,
                           'y_pred_lr': y_pred_lr,
                           'y_pred_rf': y_pred_rf})


# In[91]:


y_test_df = pd.DataFrame({'y_pred_gb': y_pred_gb,
                           'y_pred_lr': y_pred_lr,
                           'y_pred_rf': y_pred_rf})


# ###### -------------------CLUSTER  2--------------------------------

# In[92]:


final_cluster_1 = final_df[final_df['cluster']==1]


# In[93]:


final_cluster_1


# In[94]:


x_1 = final_cluster_1.drop(columns=['CLV','cluster'])


# In[95]:


x_1


# In[96]:


y_1 = final_cluster_1['CLV']


# In[97]:


xc_1train, xc_1test, yc_1train, yc_1test = train_test_split(x_1, y_1, test_size=0.2, random_state=42)


# In[98]:


lr_model = LinearRegression()
lr_model.fit(xc_1train, yc_1train)

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(xc_1train, yc_1train)


gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(xc_1train, yc_1train)


# In[99]:


def evaluate_model(model, xc_1test, yc_1test):
    y_1pred = model.predict(xc_1test)
    rmse = mean_squared_error(yc_1test, y_1pred, squared=False)
    mae = mean_absolute_error(yc_1test, y_1pred)
    r2 = r2_score(yc_1test, y_1pred)
    print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R-squared: {r2:.2f}")
    


print("\nGradient Boosting:")
evaluate_model(gb_model, xc_1test, yc_1test)


# In[100]:


print("Random Forest:")
evaluate_model(rf_model, xc_1test, yc_1test)


print("Linear Regression:")
evaluate_model(lr_model, xc_1test, yc_1test)


# #### -------------HYPER PARAMETER TUNNNING ------------

# In[101]:


gb_model = GradientBoostingRegressor(random_state=42)

param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}


# In[102]:


grid_search = GridSearchCV(estimator=gb_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(xc_1train, yc_1train)


print(f"Best hyperparameters: {grid_search.best_params_}")


# In[103]:


best_gb_model = grid_search.best_estimator_
evaluate_model(best_gb_model, xc_1test, yc_1test)


# #####  ---------- Cluster 3----------------------------------------------

# In[104]:


final_cluster_2 = final_df[final_df['cluster']==2]


# In[105]:


final_cluster_2


# In[106]:


y_2 = final_cluster_2['CLV']


# In[107]:


x_2 = final_cluster_2.drop(columns=['CLV','cluster'])


# In[108]:


xc_2train, xc_2test, yc_2train, yc_2test = train_test_split(x_2, y_2, test_size=0.2, random_state=42)


# In[109]:


lr_model = LinearRegression()
lr_model.fit(xc_2train, yc_2train)

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(xc_2train, yc_2train)


gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(xc_2train, yc_2train)


# In[110]:


def evaluate_model(model, xc_1test, yc_1test):
    y_2pred = model.predict(xc_2test)
    rmse = mean_squared_error(yc_2test, y_2pred, squared=False)
    mae = mean_absolute_error(yc_2test, y_2pred)
    r2 = r2_score(yc_2test, y_2pred)
    print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R-squared: {r2:.2f}")
    


print("\nGradient Boosting:")
evaluate_model(gb_model, xc_2test, yc_2test)


# In[111]:


print("Random Forest:")
evaluate_model(rf_model, xc_2test, yc_2test)


print("Linear Regression:")
evaluate_model(lr_model, xc_2test, yc_2test)


# In[112]:


gb_model = GradientBoostingRegressor(random_state=42)


param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}


# In[113]:


grid_search = GridSearchCV(estimator=gb_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(xc_2train, yc_2train)

print(f"Best hyperparameters: {grid_search.best_params_}")



# In[114]:


best_gb_model = grid_search.best_estimator_
evaluate_model(best_gb_model, xc_2test, yc_2test)


# ## ---------------------------Attach the customer behaviour Dataset ------------------------------------------

# In[115]:


customer_df=pd.read_csv("customer_info.csv")
customer_df.drop(columns=['Average Purcahses($) '], inplace=True)


# In[116]:


customer_df.duplicated().sum()


# In[117]:


merged_df = pd.merge(final_df, customer_df, on='CustomerID').fillna(0)


# In[118]:


merged_df.isnull().sum()


# In[119]:


non_numerical_columns = merged_df.select_dtypes(include=['object']).columns
print("non_numerical columns:", non_numerical_columns)


# In[120]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for column in non_numerical_columns:
     merged_df[column] = le.fit_transform(merged_df[column])


# In[121]:


merged_df


# In[122]:


merged_df.dropna()


# In[123]:


corr1 = merged_df.drop(columns=(['CustomerID'])).corr()


# In[124]:


plt.figure(figsize=(15, 5))
sns.heatmap(corr1, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()


# In[125]:


features = merged_df.drop(columns=(['CustomerID'])).columns


# In[126]:


n_cols = 4
n_rows = (len(features) + n_cols - 1) // n_cols  

plt.figure(figsize=(20, 5 * n_rows)) 

for i, feature in enumerate(features):
    plt.subplot(n_rows, n_cols, i + 1)
    sns.histplot(merged_df[feature], bins=30, kde=True)
    plt.title(f'Distribution of {feature}')

plt.tight_layout()
plt.show()


# In[127]:


corr=merged_df.drop(columns=["CustomerID"]).corr()


# In[128]:


n_cols = 4
n_rows = (len(features) + n_cols - 1) // n_cols  

plt.figure(figsize=(20, 5 * n_rows)) 

for i, feature in enumerate(features):
    plt.subplot(n_rows, n_cols, i + 1)
    sns.histplot(merged_df[feature], bins=30, kde=True)
    plt.title(f'Distribution of {feature}')

plt.tight_layout()
plt.show()


# In[129]:


y = merged_df['CLV']


# In[130]:


x = merged_df.drop(columns=['CLV','CustomerID'])


# In[131]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[132]:


lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)


gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)


# In[133]:


def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R-squared: {r2:.2f}")
    

print("Random Forest:")
evaluate_model(rf_model, X_test, y_test)


print("Linear Regression:")
evaluate_model(lr_model, X_test, y_test)


print("\nGradient Boosting:")
evaluate_model(gb_model, X_test, y_test)


# In[134]:


y_gb = gb_model.predict(x)


# In[135]:


y_lr = lr_model.predict(x)


# In[136]:


y_rf =rf_model.predict(x)


# In[137]:


merged_df['gb_predict'] = y_gb


# In[138]:


merged_df['lr_predict'] = y_lr


# In[139]:


merged_df['rf_predict'] = y_rf


# In[140]:


merged_df.columns


# In[141]:


visual =merged_df.drop(columns=['recency','frequency','Monetary','T','AVG_PURCHASE','Country','Gender','Age','Annual Income (k$)','Preferred Shipping Mode','Customer Satisfaction Score','Discount Usage Rate','Location ','Number Of Families  '])


# In[142]:


visual


# In[143]:


visualize = visual.head(8)

plt.figure(figsize=(20, 8))

sns.lineplot(data= visualize, x="gb_predict", y="CustomerID", label='Predicted gradient Boost', marker='o')

sns.lineplot(data=  visualize,x="lr_predict", y="CustomerID", label='predicted linear Regression', marker='o')

sns.lineplot(data= visualize, x="rf_predict", y="CustomerID", label='Predicted random forest', marker='o')

sns.lineplot(data= visualize, x="CLV", y="CustomerID", label='Actual CLV ', marker='o')

plt.title('Comparison of CLV Predictions and Actual CLV for the First 20 CustomerIDs')
plt.xlabel('CLV Value')
plt.ylabel('CustomerID')
plt.legend()


# In[144]:


merged_df

