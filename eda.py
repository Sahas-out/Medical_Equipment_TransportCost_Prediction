#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#%% 
data = pd.read_csv("train.csv")
testData = pd.read_csv("test.csv")

#%%
fig,axes = plt.subplots(nrows=(len(data.columns)//4)+1,ncols=4,figsize=(20,15))
axes = axes.flatten()
histData = data.drop(columns=['Hospital_Id','Supplier_Name','Order_Placed_Date','Delivery_Date','Hospital_Location'])
for i,col in enumerate(histData.columns):
    if(histData[col].dtype == 'object'):
        sns.countplot(x=histData[col],ax=axes[i])
    else:
       sns.histplot(x=histData[col],bins=10,kde=True,ax=axes[i])
    axes[i].set_title(f"{col}")
plt.tight_layout()
plt.savefig("hist1.png")

#%% 
boxData = data.drop(columns=['Hospital_Id','Supplier_Name','Order_Placed_Date','Delivery_Date','Hospital_Location','Equipment_Weight','Equipment_Value','Transport_Cost'])
val_col = boxData.select_dtypes(exclude=['object']).columns
fig,axes = plt.subplots(nrows=2,ncols=(len(val_col)//2)+1,figsize=(15,6))
axes = axes.flatten()
for i,col in enumerate(val_col):
    sns.boxplot(x=boxData[col],ax=axes[i])
    axes[i].set_title(f"{col}")
plt.tight_layout()
plt.savefig("boxplot2.png")

#%%
boxData = data[['Equipment_Value','Equipment_Weight','Transport_Cost']]
boxData = boxData[boxData['Transport_Cost'] > 0]
val_col = boxData.select_dtypes(exclude=['object']).columns
fig,axes = plt.subplots(nrows=2,ncols=(len(val_col)//2)+1,figsize=(15,6))
axes = axes.flatten()
for i,col in enumerate(val_col):
    sns.boxplot(x=np.log1p(boxData[col]),ax=axes[i])
    axes[i].set_title(f"{col}")
plt.tight_layout()
plt.savefig("boxplot3.png")


#%%
boxData = data.drop(columns=['Hospital_Id','Supplier_Name','Order_Placed_Date','Delivery_Date','Hospital_Location'])
boxData = boxData[boxData['Transport_Cost'] > 0]
obj_cols = boxData.select_dtypes(include=['object']).columns
fig,axes = plt.subplots(nrows=2,ncols=(len(obj_cols)//2)+1,figsize=(15,6))
axes = axes.flatten()
for i,col in enumerate(obj_cols):
    sns.boxplot(x=np.log1p(boxData['Transport_Cost']),hue=boxData[col],ax=axes[i])
    axes[i].set_title(f"{col}")
plt.tight_layout()
plt.savefig("boxplot1.png")


#%%
corr = data[data.select_dtypes(exclude=['object']).columns.tolist()].corr()
plt.figure(figsize=(20,15))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.savefig("correlation.png")

#%%
from scipy.stats import f_oneway
obj_cols = data.drop(columns=['Hospital_Id','Hospital_Location']).select_dtypes(include=['object']).columns
for col in obj_cols:
    groups = [group_df['Transport_Cost'].values for name,group_df in data.groupby(col)]
    f_stat,p_value = f_oneway(*groups)
    print(f"for {col} f_stat:{f_stat} p_value:{p_value}")

#%%
#Cross bordeer shipping and urgent_Shipping and Fragile Equipment Order_Placed_date - Delivery Date
fig,axes = plt.subplots(nrows=1,ncols =2,figsize=(15,6))
axes = axes.flatten()
sns.histplot(x=data['CrossBorder_Shipping'],hue=data['Transport_Method'],ax=axes[0],multiple='dodge')
sns.histplot(x=data['Urgent_Shipping'],hue=data['Transport_Method'],ax=axes[1],multiple='dodge')
plt.tight_layout()
plt.savefig("fig4")

#%%
data['Delivery_Time'] = (data['Delivery_Date']-data['Order_Placed_Date']).dt.days
boxData = data[data['Delivery_Time'] > 0]
sns.boxplot(x=boxData['Delivery_Time'],hue=boxData['Transport_Method'])
plt.savefig('fig5')

#%%
sns.boxplot(
    x = np.log1p(data['Equipment_Weight'] / (data['Equipment_Height'] * data['Equipment_Width'])),
   hue = data['Equipment_Type']
)
plt.savefig('fig100')

#%% 
sns.boxplot(x=data['Equipment_Value'],hue=data['Rural_Hospital'])
plt.savefig("fig800")

#%%
data_tr = data[data['Transport_Cost'] > 0].copy()
data_tr['Transport_Cost'] = np.log1p(data_tr['Transport_Cost'])

#%%
data['Order_Placed_Date'] = pd.to_datetime(data['Order_Placed_Date'])
data['Delivery_Date'] = pd.to_datetime(data['Delivery_Date'])

# Add time features
data['Order_Month'] = data['Order_Placed_Date'].dt.month
data['Order_DayOfWeek'] = data['Order_Placed_Date'].dt.dayofweek
data['Order_Year'] = data['Order_Placed_Date'].dt.year

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Lineplot: log(Transport_Cost) over Order_Placed_Date
sns.lineplot(
    # x=data['Order_Placed_Date'],
    # y=np.log1p(data[data['Transport_Cost'] > 0]['Transport_Cost']),
    x = 'Order_Placed_Date',
    y = 'Transport_Cost',
    data = data_tr,
    ax=axes[0, 0]
)
axes[0, 0].set_title("Log(Transport_Cost) over Order_Placed_Date")

# Histogram: mean Transport_Cost by Month
sns.barplot(
    x='Order_Month',
    y='Transport_Cost',
    data=data,
    estimator=np.mean,
    ax=axes[0, 1]
)
axes[0, 1].set_title("Mean Transport_Cost by Month")

# Histogram: mean Transport_Cost by Day of Week
sns.barplot(
    x='Order_DayOfWeek',
    y='Transport_Cost',
    data=data,
    estimator=np.mean,
    ax=axes[1, 0]
)
axes[1, 0].set_title("Mean Transport_Cost by Day of Week")

# Histogram: mean Transport_Cost by Year
sns.barplot(
    x='Order_Year',
    y='Transport_Cost',
    data=data,
    estimator=np.mean,
    ax=axes[1, 1]
)
axes[1, 1].set_title("Mean Transport_Cost by Year")

plt.tight_layout()
plt.savefig("transport_cost_vs_date_analysis.png")

#%%
import re

def parse_location(location_str):
    """Parse hospital location to extract state, zip, and military status"""
    if pd.isna(location_str) or location_str == 'Missing':
        return 'Missing', 'Missing', 0

    # Check for military addresses (APO, FPO, DPO with AA, AE, AP)
    is_military = 1 if re.search(r'\b(APO|FPO|DPO)\b', str(location_str), re.IGNORECASE) else 0
    is_military = is_military or (1 if re.search(r'\b(AA|AE|AP)\b', str(location_str)) else 0)

    # Extract state (2 letter code)
    state_match = re.search(r'\b([A-Z]{2})\b\s+\d{5}', str(location_str))
    if state_match:
        state = state_match.group(1)
    else:
        state = 'Missing'

    # Extract zip code (5 digits)
    zip_match = re.search(r'\b(\d{5})\b', str(location_str))
    if zip_match:
        zip_code = zip_match.group(1)
    else:
        zip_code = 'Missing'

    return state, zip_code, is_military

# Apply parser to Hospital_Location column
data[['State', 'Zipcode', 'IsMilitary']] = data['Hospital_Location'].apply(
    lambda x: pd.Series(parse_location(x))
)
data_tr[['State', 'Zipcode', 'IsMilitary']] = data_tr['Hospital_Location'].apply(
    lambda x: pd.Series(parse_location(x))
)

#%%
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
sns.boxplot(
    x='Transport_Cost',
    y='IsMilitary',
    data=data_tr,
    orient='h',
    ax=axes[0]
)
axes[0].set_yticklabels(['Non-Military', 'Military'])
axes[0].set_title("Transport Cost Distribution by Military Status")

# Barplot: Mean Transport_Cost by State
sns.barplot(
    x='State',
    y='Transport_Cost',
    data=data,
    estimator=np.mean,
    ax=axes[1]
)
axes[1].set_title("Mean Transport Cost by State")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig("transport_cost_military_state.png")

#%%
cols_to_impute = ['Transport_Method', 'Rural_Hospital', 'Equipment_Type']
data[cols_to_impute] = data[cols_to_impute].fillna('Unknown')

