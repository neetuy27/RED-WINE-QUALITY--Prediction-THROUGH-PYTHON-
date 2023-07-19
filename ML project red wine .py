#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import warnings 
warnings.filterwarnings('ignore')



# In[4]:


df=pd.read_csv('wineQualityReds.csv')


# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


df.sample(5)


# In[8]:


df.shape


# In[9]:


df.info()


# In[12]:


df.isnull().sum()


# In[13]:


df.shape


# In[17]:


duplicate=df.duplicated()
print(duplicate.sum())
df[duplicate]


# In[18]:


df.dropna()


# In[19]:


print(df.quality.value_counts())


# In[31]:


sns.countplot(df['quality'])
plt.grid()
plt.show()


# In[24]:


df.corr()


# In[30]:


corr= df.corr()
plt.figure(figsize=(30,20))
sns.heatmap(corr,annot=True,cmap='coolwarm')


# In[32]:


target_name='quality'
y=df[target_name]
x=df.drop(target_name,axis=1)


# In[33]:


x.head()


# In[35]:


x.shape


# In[36]:


y.head()


# In[37]:


y.shape


# In[38]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_res = sc.fit_transform(x) 


# In[39]:


x.head()


# In[40]:


x_res


# In[41]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data = pd.DataFrame()
vif_data["vif"] =[variance_inflation_factor(x_res,i) for i in range(x_res.shape[1])]
vif_data["Features"]=x.columns
vif_data


# In[42]:


x_res.shape


# In[43]:


x1=x.drop(['residual.sugar','density'],axis=1)
x1.shape


# In[52]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
(scaler.fit(x1))
rescaledx = scaler.transform(x1)


# In[53]:


rescaledx.shape


# In[54]:


y.value_counts()


# In[56]:


sns.countplot(df['quality'])
plt.grid()
plt.show()


# In[57]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(rescaledx,y,test_size=0.2,random_state=7)


# In[58]:


x_train.shape,y_train.shape


# In[59]:


x_test.shape,y_test.shape


# In[60]:


from sklearn.tree import DecisionTreeClassifier
dt= DecisionTreeClassifier()
dt.fit(x_train,y_train)
dt.train_pred=dt.predict(x_train)
dt_test_pred=dt.predict(x_test)


# In[61]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[79]:


print('Train Accuracy:',accuracy_score(y_train,dt.train_pred)*100)


# In[81]:


print('Accuracy Score:',accuracy_score(y_test,dt_test_pred)*100)


# In[84]:


print(confusion_matrix(y_test,dt_test_pred))


# In[85]:


print(classification_report(y_test,dt_test_pred,digits=4))


# In[89]:


(0.0000+0.1000+0.6667+0.6912+0.5122+0.3333)/6


# In[87]:


from sklearn.metrics import precision_score,recall_score,f1_score,classification_report,confusion_matrix
print("precision score of macro is:",round(precision_score(y_test,dt_test_pred,average='macro')*100,2))
print("precision score of micro is:",round(precision_score(y_test,dt_test_pred,average='micro')*100,2))
print("precision score of weighted is:",round(precision_score(y_test,dt_test_pred,average='weighted')*100,2))
            


# In[88]:


print("recall_score of macro is:",round(recall_score(y_test,dt_test_pred,average='macro')*100,2))
print("recall_score of micro is:",round(recall_score(y_test,dt_test_pred,average='micro')*100,2))
print("recall_score of weighted is:",round(recall_score(y_test,dt_test_pred,average='weighted')*100,2))    
      


# In[90]:


print('f1_score of macro:',round(f1_score(y_test,dt_test_pred,average='macro')*100,2))
print('f1_score of micro:',round(f1_score(y_test,dt_test_pred,average='micro')*100,2))
print('f1_score of weighted:',round(f1_score(y_test,dt_test_pred,average='weighted')*100,2))




# In[91]:


df.head()


# In[103]:


input_data = (1,7.4,0.70,0.00,1.9,0.07,11.0,34.0,0.9978,3.51)
input_data_as_numpy_array=np.asarray(input_data)

input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

prediction = dt.predict(input_data_reshaped)

print(prediction)



# In[111]:


from sklearn.ensemble import RandomForestClassifier
clf= RandomForestClassifier(random_state=1)

clf.fit(x_train,y_train)

clf_train_pred = clf.predict(x_train)

clf_test_pred = clf.predict(x_test)


# In[112]:


print('Train_accuracy:',accuracy_score(y_train,clf_train_pred)*100)


# In[114]:


print('Accuracy Score:',accuracy_score(y_test,clf_test_pred)*100)


# In[115]:


print(confusion_matrix(y_test,clf_test_pred))


# # 

# In[117]:


print("recall_score of macro is:",round(recall_score(y_test,clf_test_pred,average='macro')*100,2))
print("recall_score of micro is:",round(recall_score(y_test,clf_test_pred,average='micro')*100,2))
print("recall_score of weighted is:",round(recall_score(y_test,clf_test_pred,average='weighted')*100,2))    
      


# In[118]:


print('f1_score of macro:',round(f1_score(y_test,clf_test_pred,average='macro')*100,2))
print('f1_score of micro:',round(f1_score(y_test,clf_test_pred,average='micro')*100,2))
print('f1_score of weighted:',round(f1_score(y_test,clf_test_pred,average='weighted')*100,2))


# In[127]:


from sklearn import svm


# In[128]:


clf = svm.SVC(probability=True)
clf.fit(x_train,y_train)


# In[143]:


clf_train_pred=clf.predict(x_train)
clf_train_pred=clf.predict(x_test)


# In[149]:


print('Train Accuracy:',accuracy_score(y_test,clf_train_pred)*100)


# In[153]:


print('Accuracy Score:',accuracy_score(y_test,clf_test_pred)*100)


# In[154]:


print(confusion_matrix(y_test,clf_test_pred))


# In[155]:


print("precision score of macro is:",round(precision_score(y_test,clf_test_pred,average='macro')*100,2))
print("precision score of micro is:",round(precision_score(y_test,clf_test_pred,average='micro')*100,2))
print("precision score of weighted is:",round(precision_score(y_test,clf_test_pred,average='weighted')*100,2))
            


# In[156]:


print("recall_score of macro is:",round(recall_score(y_test,clf_test_pred,average='macro')*100,2))
print("recall_score of micro is:",round(recall_score(y_test,clf_test_pred,average='micro')*100,2))
print("recall_score of weighted is:",round(recall_score(y_test,clf_test_pred,average='weighted')*100,2))    
      


# In[157]:


print('f1_score of macro:',round(f1_score(y_test,clf_test_pred,average='macro')*100,2))
print('f1_score of micro:',round(f1_score(y_test,clf_test_pred,average='micro')*100,2))
print('f1_score of weighted:',round(f1_score(y_test,clf_test_pred,average='weighted')*100,2))


# In[ ]:




