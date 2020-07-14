#!/usr/bin/env python
# coding: utf-8

# In[322]:


## Importing Data 
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats  as stats

Males = pd.read_csv("MA_Exer_PikesPeak_Males", sep='\t', encoding='latin-1')
Males.columns = ["Place", "Div/Tot", "ID", "Name", "Age", "Hometown", "Gun Time", "Net Time", "Pace"]
Males


Females = pd.read_csv("MA_Exer_PikesPeak_Females.txt", sep='\t', encoding='ASCII')
Females.columns = ["Place", "Div/Tot", "ID", "Name", "Age", "Hometown", "Gun Time", "Net Time", "Pace"]
Females

## ASSIGN GENDER COLUMN TO BOTH DATA FRAMES
Females["Gender"] = 'F'
Males["Gender"] = 'M'

## APPEND BOTH DATA FILES
Data = Males.append(Females, ignore_index=True)


################# CLEANING DATA ########################################
## CHECK FOR NULL VALUES and Data Types
Data.dtypes
Data.isnull().any()  #Null in DIV TOT and Age 
Data.describe()


## REMOVE CHARACTERS FROM TIME COLUMNS
Data['Gun Time'] = Data['Gun Time'].str.replace(r'[a-zA-Z]', '')
Data['Net Time'] = Data['Net Time'].str.replace(r'[*,#]', '')
Data['Pace'] = Data['Pace'].str.replace(r'[a-zA-Z]', '')



## SEPARATE DIV/TOT VARIABLE TO VERIFY DIVISIONS AND TOTALS
Data[['Division','Total']] = Data['Div/Tot'].str.split("/",expand=True,)
print(Data)


# In[323]:



################# Working with Time variables #####################


## Object to Time Delta ##
Data['Gun Time'] = pd.to_timedelta(
    np.where(Data['Gun Time'].str.count(':') == 1, '00:' + Data['Gun Time'], Data['Gun Time']))

Data['Net Time'] = pd.to_timedelta(
    np.where(Data['Net Time'].str.count(':') == 1, '00:' + Data['Net Time'], Data['Net Time']))

Data['Pace'] = pd.to_timedelta(
    np.where(Data['Pace'].str.count(':') == 1, '00:' + Data['Pace'], Data['Pace']))

## Create Minute Variables 
Data['Net Minutes'] =Data['Net Time'] / np.timedelta64(1, 'm')
Data['Gun Minutes'] =Data['Gun Time'] / np.timedelta64(1, 'm')
Data['Pace Minutes'] =Data['Pace'] / np.timedelta64(1, 'm')

## Create Net /GUN Time Difference Variable
Data['Difference Minutes'] = Data['Gun Minutes'] - Data['Net Minutes']

###################### Working with Place Variable #########################

## Categorize Place variable to Make for Easier Data Visualization
bins= [0,50,100,200,300,400,500,600,700,2000]
labels = ['1-49','50-99','100s','200s','300s','400s','500s','600s', '700+']
Data['Places'] = pd.cut(Data['Place'], bins=bins, labels=labels, right=False)

#########################  Hometown Variable ##########################

## Clean hometown

Data.loc[(Data['Hometown'].isnull()), 'Hometown'] = np.nan
HomeCounts = Data['Hometown'].value_counts(dropna=True)

print(HomeCounts)


# In[324]:


####################### Working with Age Variable ##########################
# Mean Age for Null Values after examining data groups

Male_Age = Data['Age'][(Data['Total'] == '28') & (Data['Gender'] == 'M')].mean()
Female_Age = Data['Age'][(Data['Total'] == '15') & (Data['Gender'] == 'F')].mean()



## Fill in nulls with na 
Data.loc[(Data['Age'] <=1) | (Data['Age'].isnull()), 'Age'] = np.nan

## Fill in Nulls with 0 to group age for group transform
Data.loc[(Data['Total'].isnull()), 'Total'] = 0



## Transform group to fill na age to mean of gender and division
Data['Age'] = Data.groupby(['Gender', 'Total'])['Age'].transform(lambda x: x.fillna(np.mean(x)))



##### Check for Totals to match number in Divisions #####
Data['Total'] = Data['Total'].astype(str).astype(int)
df1 = Data.groupby(['Gender','Total']).agg({'Total':'size'})

TotCounts = Data['Total'].value_counts(dropna=True)





##ASSIGNING Age Groups And New Divisions
bins= [0,15,20,30,40,50,60,70,80,90]
labels = ['0-14','15-19','20-29','30-39','40-49','50-59','60-69','70-79','80-89']
Data['AgeGroup'] = pd.cut(Data['Age'], bins=bins, labels=labels, right=False)
print (Data)


bins= [0,15,20,30,40,50,60,70,80,90]
labels = [1,2,3,4,5,6,7,8,9]
Data['New_Division'] = pd.cut(Data['Age'], bins=bins, labels=labels, right=False)


#Counting Male and Female
DataGender = Data.groupby(['Gender','New_Division'])['ID'].nunique()

 

## To detect split between city with two names Change length 1 or 2 
Data[['City','State']] = Data['Hometown'].str.rsplit(" ", 1,expand=True)
mask = Data['State'].str.len() > 2
df = Data.loc[mask]
print(df)


## Correcting City
Data.loc[Data['State'] == 'Bethesda', 'City'] = "North Bethesda"
Data.loc[Data['State'] == 'Grov', 'City'] = "Washington Grov"
Data.loc[Data['State'] == 'Heights', 'City'] = "Capitol Heights"
Data.loc[Data['State'] == 'Park', 'City'] = "University Park"
Data.loc[Data['State'] == 'Station', 'City'] = "Fairfax Station"
Data.loc[Data['State'] == 'Vill', 'City'] = "Montgomery Vill"

# Changing state to null 
Data.loc[Data['State'] == 'Bethesda', 'State'] = "."
Data.loc[Data['State'] == 'Grov', 'State'] = "."
Data.loc[Data['State'] == 'Heights', 'State'] = "."
Data.loc[Data['State'] == 'Park', 'State'] = "."
Data.loc[Data['State'] == 'Station', 'State'] = "."
Data.loc[Data['State'] == 'Vill', 'State'] = "."

## Changing Unknown and Wash and Welcome to Null 
Data.loc[Data['City'] == 'Unknown', 'City'] = "."
Data.loc[Data['City'] == 'Wash', 'City'] = "."
Data.loc[Data['City'] == 'Welcome', 'City'] = "."


#Correcting state code 
Data.loc[Data['City'] == 'Silver Spring', 'State'] = "MD"
Data.loc[Data['City'] == 'Ellicott City', 'State'] = "MD"
Data.loc[Data['City'] == 'North Potomac', 'State'] = "MD"
Data.loc[Data['City'] == 'Pembroke Pines', 'State'] = "FL"
Data.loc[Data['City'] == 'Potomac Falls', 'State'] = "VA"
Data.loc[Data['City'] == 'Marriottsville', 'State'] = "VA"
Data.loc[Data['City'] == 'North Bethesda', 'State'] = "MD"

## City name error changes
Data.loc[Data['City'] == 'Ss', 'City'] = "Silver Spring"
Data.loc[Data['City'] == 'Silverspring', 'City'] = "Silver Spring"
Data.loc[Data['City'] == 'Silver Spring ', 'City'] = "Silver Spring"
Data.loc[Data['City'] == 'Rockvile', 'City'] = "Rockville"
Data.loc[Data['City'] == 'Rockvilee', 'City'] = "Rockville"
Data.loc[Data['City'] == 'Potomc', 'City'] = "Potomac"
Data.loc[Data['City'] == 'Poollesville', 'City'] = "Poolesville"
Data.loc[Data['City'] == 'North', 'City'] = "North Bethesda"
Data.loc[Data['City'] == 'N.potomac', 'City'] = "North Potomac"
Data.loc[Data['City'] == 'N. Potomac', 'City'] = "North Potomac"
Data.loc[Data['City'] == 'N Potomac', 'City'] = "North Potomac"
Data.loc[Data['City'] == 'N Bethesda', 'City'] = "North Bethesda"
Data.loc[Data['City'] == 'N. Bethesda', 'City'] = "North Bethesda"
Data.loc[Data['City'] == 'My Airy', 'City'] = "Mount Airy"
Data.loc[Data['City'] == 'Mt. Airy', 'City'] = "Mount Airy"
Data.loc[Data['City'] == 'Mongtomery', 'City'] = "Montgomery"
Data.loc[Data['City'] == 'Germantownt', 'City'] = "Germantown"
Data.loc[Data['City'] == 'Gertmantown', 'City'] = "Germantown"
Data.loc[Data['City'] == 'Gatihersburg', 'City'] = "Mount Airy"
Data.loc[Data['City'] == 'Gaithersbur', 'City'] = "Mount Airy"
Data.loc[Data['City'] == 'Gaithersbug', 'City'] = "Montgomery"
Data.loc[Data['City'] == 'Ellcott City', 'City'] = "Ellicott City"
Data.loc[Data['City'] == 'Dunkrirk', 'City'] = "Dunkirk"
Data.loc[Data['City'] == 'Chevy Chas', 'City'] = "Chevy Chase"
Data.loc[Data['City'] == 'Bathesda', 'City'] = "Bethesda"



# In[325]:


## Check cleaning 
CityCounts = Data['City'].value_counts(dropna=True)
StateCounts = Data['State'].value_counts(dropna=True)
print(Data.dtypes)
Clean = Data.isnull().any()
print(Clean)


# In[319]:


## QUESTION 1 : Mean, Median, Mode & Range by Gender
csv3 = Data[['Net Minutes', 'Gun Minutes','Pace Minutes','Place', 'Gender']].copy()
MeanGender = csv3.groupby(['Gender']).mean()
MedianGender = csv3.groupby(['Gender']).median()
ModeGender = csv3.groupby(['Gender']).apply(lambda x: x.mode().iloc[0])

#Range = Max- Min 
mm1 = csv3.groupby(by='Gender').agg({'Net Minutes': ['min','max']})
mm2 = csv3.groupby(by='Gender').agg({'Gun Minutes': ['min','max']})
mm3 = csv3.groupby(by='Gender').agg({'Pace Minutes': ['min','max']})
                                     
print(MeanGender)
print(MedianGender)
print(ModeGender)
print(mm1)
print(mm2)
print(mm3)


# In[326]:


#### QUESTION 2 : Difference between Net and Gun Time Results 


# In[327]:


## Spearman's rho correlation since Place is a ranked variable
sp, pv = stats.spearmanr(Data['Place'], Data['Difference Minutes'])
spear = print('Spearmans rho correlation: %.3f p-value %.3f'% (sp, pv))


print(spear)


# In[328]:


sns.set(font_scale = 1)

with sns.color_palette("Set2"):
    g = sns.lineplot(x="Places", y="Difference Minutes", hue="Gender", palette="Set1", data=Data).set_title("Figure 1: 2006 Pike Peak 10k Race Results\n Difference between Gun/Net Times by Race Place and Gender");


# In[329]:


#### QUESTION 3 : Chris Doe Net Time Difference to be in top 10 Percentile  ####
    
Check_Chris = Data.query('Name == "Chris Doe"')
p_10 = Data['Net Minutes'].quantile(0.1)
p_90 = Data['Net Minutes'].quantile(0.9)

## Only Males Division
csv = Data[(Data['New_Division'] == 5)]
onlymales = csv[(csv['Gender'] == 'M')]
onlymalescsv = onlymales[['Net Minutes', 'New_Division']].copy()

## Males and Females
csv1 = csv[['Net Minutes', 'New_Division','AgeGroup','Name', 'Gender']].copy()
csv2 = csv[['Net Minutes', 'New_Division']].copy()



# Create box plot and point for chris doe 
sns.set(font_scale = 1)
Chart2 = sns.swarmplot(x="AgeGroup", y="Net Minutes", hue='Name', data=csv1.query('Name == "Chris Doe"'),  color="1", palette ="Set1")
Chart1 = sns.boxplot(x="AgeGroup", y="Net Minutes", hue='Gender', data=csv1.query('New_Division ==5'), palette = "Set2").set_title("Figure 2: Chris Doe: Comparison to Same Division (Ages 40-49)\n (n=698)")

# Get Top 10 Percentile Threshold
p_10 = csv2.groupby(['New_Division']).quantile(0.1)  #males and Females
p_10m = onlymalescsv.groupby(['New_Division']).quantile(0.1)  #Only Males

# Get counts 
ChrisDivision = csv1.groupby(by='Gender').agg({'Net Minutes': ['count']})

print(ChrisDivision)
print(p_10) ## Overall males and females same age group
print(p_10m)## top 10 percentile only males (same division as chris)
# Get Chris Doe's Net Minute Time 
print(csv1.query('Name == "Chris Doe"'))



# In[330]:


#### QUESTION 4 : Compare Results by Division ################################


# In[265]:


sns.set_style("whitegrid")
sns.set(font_scale = 1)
box_plot = sns.boxplot(x="AgeGroup", y="Net Minutes", data=Data).set_title("Figure 3: 2006 Pike Peak 10k Race Results: \n By Each Division/Age Group \n")

ax = box_plot.axes
lines = ax.get_lines()
categories = ax.get_xticks()

for cat in categories:
    # every 4th line at the interval of 6 is median line
    # 0 -> p25 1 -> p75 2 -> lower whisker 3 -> upper whisker 4 -> p50 5 -> upper extreme value
    y = round(lines[4+cat*6].get_ydata()[0],1) 

    ax.text(
        cat, 
        y, 
        f'{y}', 
        ha='center', 
        va='center', 
        fontweight='semibold', 
        size=9,
        color='white',
        bbox=dict(facecolor='#445A64'))

box_plot.figure.tight_layout()


# In[331]:


sns.set_style("whitegrid")
sns.set(font_scale = 1)
box_plot2 = sns.boxplot(x="AgeGroup", y="Net Minutes", hue='Gender', data=Data).set_title("Figure 4: 2006 Pike Peak 10k Race Results: \n By Each Division/Age Group and Gender \n")


# In[332]:


## Not using Fig3 = sns.distplot(Data['Pace Minutes'],hist=True,bins=100)


# In[333]:


#Not using Fig4 = sns.catplot(x="AgeGroup", y="Net Minutes", hue="Gender",palette="Set2", dodge=True, kind="swarm", height=7,  data=Data)


# In[334]:


## Creating Dimensions
fig_dims = (20,8)
fig, ax = plt.subplots(figsize=fig_dims)
sns.set(font_scale = 1.5)

with sns.color_palette("husl"):
    g = sns.lineplot(x="Age", y="Net Minutes", hue="AgeGroup", style="Gender", ax=ax, data=Data).set_title("Figure 5: 2006 Pike Peak 10k Race Results: \n By Divisions/Age Group" );


plt.setp(ax.get_legend().get_texts(), fontsize='14') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='16') 


# In[ ]:




