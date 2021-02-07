import numpy as np
import pandas as pd

from sklearn.metrics import auc
from scipy import stats
from math import e



class BioquivalenceMathsModel:

    def get_auc(self, x: np.array, y: np.array) -> float:
        return auc(x, y)

    def get_log_array(self, x: np.array) -> np.array:
        return np.log(x)

    def get_kstest(self, x: np.array) -> tuple:
        x = (x-np.mean(x))/np.std(x)
        return stats.kstest(x, 'norm')

    def get_shapiro(self, x: np.array) -> tuple:
        return stats.shapiro(x)
    
    def get_f(self, x: np.array, y: np.array) -> tuple:
        return stats.f_oneway(x, y)

    def get_levene(self, x: np.array, y: np.array) -> tuple:
        lx = []
        for i in range(x.size):
            lx.append(float(x[i]))
        ly = []
        for i in range(y.size):
            ly.append(float(y[i]))
        return stats.levene(lx, ly)

    def get_mannwhitneyu(self, x: np.array, y: np.array) -> tuple:
        return stats.mannwhitneyu(x, y)

    def get_wilcoxon(self, x: np.array, y: np.array) -> tuple:
        lx = []
        for i in range(x.size):
            lx.append(float(x[i]))
        ly = []
        for i in range(y.size):
            ly.append(float(y[i]))
        return stats.wilcoxon(lx, ly)

    def get_k_el(self, x: np.array, y: np.array) -> float:
        return np.polyfit(x[-3:],y[-3:], deg=1)[0]

    def get_anova(self, x: np.array, y: np.array, z: np.array) -> tuple:
        dff = pd.read_csv('f_critical.csv',index_col=0)
        iind=0
        cind=0
        if x.size > 120:
            iind = -1
            cind = -1
        else:
            df1 = 1
            df2 = z.size - 2
            ind = np.array(dff.index)
            ind = ind[:-1]
            col = np.array(dff.columns)
            col = col[:-1]
            col = col.astype("float")
            ind = abs(ind-df2)
            col = abs(col-df1)
            iind = np.argmin(ind)#ИНДЕКСЫ, использовать iloc (возвращает индекс какой надо, -1 не делать)
            cind = np.argmin(col)
        ssb = x.size*(np.mean(x)-np.mean(z))**2 + y.size*(np.mean(y)-np.mean(z))**2
        sse = np.sum((x-np.mean(x))**2)+np.sum((y-np.mean(y))**2)
        sst = np.sum((z-np.mean(z))**2)
        data = {'SS':[ssb,sse,sst],'df':[1, z.size-2, z.size-1],'MS':[ssb, sse/(z.size-2), '-'],
        'F':[ssb/(sse/(z.size-2)), '-', '-'], 'F крит.':[dff.iloc[iind,cind], '-', '-']}
        df = pd.DataFrame(data)
        res = ssb/(sse/(z.size-2)) < dff.iloc[iind,cind]
        return df, res

    def get_oneside(self, x: np.array, y: np.array, df: pd.DataFrame) -> tuple:
        dft = pd.read_csv('t_critical.csv',index_col=0)
        if x.size + y.size > 31:
            df1 = dft.loc['inf']
        else:
            df1 = dft.loc[x.size + y.size - 2]
        left = float(np.mean(x) - np.mean(y) - df1*(4*df.iloc[1, 2]/(x.size + y.size))**(1/2))
        right = float(np.mean(x) - np.mean(y) + df1*(4*df.iloc[1, 2]/(x.size + y.size))**(1/2))
        return left, right

    def create_auc(self, df: pd.DataFrame) ->np.array:
        time = np.array(df.columns)
        aucс = df.apply(lambda row: pd.Series({'auc':auc(time, row)}),axis=1)
        return np.array(aucс)


    def create_auc_infty(self, df: pd.DataFrame) -> np.array:
        time = np.array(df.columns)
        aucс = df.apply(lambda row: pd.Series({'auc':auc(time, row)}),axis=1)
        auuc = np.array(aucс)
        k_el_divided = df.apply(lambda row: pd.Series({'k_el_divided':self.get_k_el(time, row)/row.iloc[-1]}),axis=1)
        k_el_divided = np.array(k_el_divided)
        return auuc + k_el_divided

    def isnormal(self) -> bool:
        if (self.check_normal == 0 and self.kstest_t[1] > self.alpha and 
            self.kstest_r[1] > self.alpha):
            return True
        elif (self.check_normal == 1 and self.shapiro_t[1] > self.alpha and 
            self.shapiro_r[1] > self.alpha):
            return True
        elif (self.check_normal == 2 and self.shapiro_t[1] > self.alpha and 
            self.shapiro_r[1] > self.alpha and self.kstest_t[1] > self.alpha and 
            self.kstest_r[1] > self.alpha):
            return True
        return False

    def isdifferent(self) -> bool:
        if (self.check_diff == 0 and self.mannwhitneyu[1] > self.alpha):
            return True
        elif (self.check_diff == 1 and self.wilcoxon[1] > self.alpha):
            return True
        elif (self.check_diff == 2 and self.mannwhitneyu[1] > self.alpha and 
            self.wilcoxon[1] > self.alpha):
            return True
        return False

    def isuniformly(self) -> bool:
        if (self.check_uniformity == 0 and self.f[1] > self.alpha):
            return True
        elif (self.check_uniformity == 1 and self.levene[1] > self.alpha):
            return True
        elif (self.check_uniformity == 2 and self.f[1] > self.alpha and 
            self.levene[1] > self.alpha):
            return True
        return False
    
    def log_auc(self):
        self.auc_log = True
        self.auc_t = self.get_log_array(self.auc_t)
        self.auc_r = self.get_log_array(self.auc_r)
        self.auc = np.concatenate((self.auc_t,self.auc_r))
        

    def __init__(self, settings: dict):
        self.concentration_t = settings.get('concentration_t',0)
        self.concentration_r = settings.get('concentration_r',0)
        self.alpha = 0.05
        self.auc_t = settings.get('auc_t',0)
        self.auc_r = settings.get('auc_r',0)
        self.auc_t_notlog = settings.get('auc_t',0)
        self.auc_r_notlog = settings.get('auc_r',0)
        self.auc_log = False
        if type(self.auc_t) == pd.DataFrame:
            self.auc = np.concatenate((self.auc_t,self.auc_r))
        else:
            self.auc = 0
        self.plan = settings['plan_bio']#0 - параллельный, 1 - перекрестный
        if type(self.concentration_t) == pd.DataFrame:
            self.auc_t_infty = 0
            self.auc_r_infty = 0
            self.auc_t_infty_log = 0
            self.auc_r_infty_log = 0
        self.check_normal = settings['check_normal_bio']#0 - ks, 1 - shapiro, 2 - both
        self.check_diff = settings['check_diff_bio']#0 - mannwhitneyu, 1 - wilcoxon, 2 - both
        self.check_uniformity = settings['check_uniformity_bio']#0 - f, 1 - levene, 2 - both
        self.kstest_t = 0
        self.kstest_r = 0
        self.shapiro_t = 0
        self.shapiro_r = 0
        self.mannwhitneyu = 0
        self.wilcoxon = 0
        self.f = 0
        self.levene = 0
        self.anova = 0
        self.oneside = 0
        
        
    def run_bio_model(self):
        if self.plan == 0:
            if type(self.concentration_t) == pd.DataFrame:
                self.auc_t = self.create_auc(self.concentration_t)
                self.auc_r = self.create_auc(self.concentration_r)
                self.auc_t_notlog = self.auc_t
                self.auc_r_notlog = self.auc_r
                self.auc = np.concatenate((self.auc_t,self.auc_r))
                self.auc_t_infty = self.create_auc_infty(self.concentration_t)
                self.auc_r_infty = self.create_auc_infty(self.concentration_r)
                self.auc_t_infty_log = self.get_log_array(self.auc_t_infty)
                self.auc_r_infty_log = self.get_log_array(self.auc_r_infty)
            if self.check_normal == 0:
                self.kstest_t = self.get_kstest(self.auc_t)#колмогоров только для стандартного
                self.kstest_r = self.get_kstest(self.auc_r)
                if (self.kstest_t[1] <= self.alpha or 
                    self.kstest_r[1] <= self.alpha):
                    self.log_auc()
                    self.kstest_t = self.get_kstest(self.auc_t)
                    self.kstest_r = self.get_kstest(self.auc_r)
            elif self.check_normal == 1:
                self.shapiro_t = self.get_shapiro(self.auc_t)
                self.shapiro_r = self.get_shapiro(self.auc_r)
                if (self.shapiro_t[1] <= self.alpha or 
                    self.shapiro_r[1] <= self.alpha):
                    self.log_auc()
                    self.shapiro_t = self.get_shapiro(self.auc_t)
                    self.shapiro_r = self.get_shapiro(self.auc_r)
            else:
                self.kstest_t = self.get_kstest(self.auc_t)
                self.kstest_r = self.get_kstest(self.auc_r)
                self.shapiro_t = self.get_shapiro(self.auc_t)
                self.shapiro_r = self.get_shapiro(self.auc_r)
                if (self.kstest_t[1] <= self.alpha or 
                    self.kstest_r[1] <= self.alpha or
                   self.shapiro_t[1] <= self.alpha or 
                    self.shapiro_r[1] <= self.alpha):
                    self.log_auc()
                    self.kstest_t = self.get_kstest(self.auc_t)
                    self.kstest_r = self.get_kstest(self.auc_r)
                    self.shapiro_t = self.get_shapiro(self.auc_t)
                    self.shapiro_r = self.get_shapiro(self.auc_r)
            if self.check_diff == 0:
                self.mannwhitneyu = self.get_mannwhitneyu(self.auc_t, self.auc_r)
                if self.mannwhitneyu[1] <= self.alpha and self.auc_log==False:
                    self.log_auc()
                    self.mannwhitneyu = self.get_mannwhitneyu(self.auc_t, self.auc_r)
            elif self.check_diff == 1:
                self.wilcoxon = self.get_wilcoxon(self.auc_t, self.auc_r)
                if self.wilcoxon[1] <= self.alpha and self.auc_log==False:
                    self.log_auc()
                    self.wilcoxon = self.get_wilcoxon(self.auc_t, self.auc_r)
            else:
                self.mannwhitneyu = self.get_mannwhitneyu(self.auc_t, self.auc_r)
                self.wilcoxon = self.get_wilcoxon(self.auc_t, self.auc_r)
                if self.auc_log==False and (self.mannwhitneyu[1] <= self.alpha or
                                           self.wilcoxon[1] <= self.alpha):
                    self.log_auc()
                    self.mannwhitneyu = self.get_mannwhitneyu(self.auc_t, self.auc_r)
                    self.wilcoxon = self.get_wilcoxon(self.auc_t, self.auc_r)
            if self.check_uniformity == 0:
                self.f = self.get_f(self.auc_t, self.auc_r)
                if self.f[1] <= self.alpha and self.auc_log==False:
                    self.log_auc()
                    self.f = self.get_f(self.auc_t, self.auc_r)
            elif self.check_uniformity == 1:
                self.levene = self.get_levene(self.auc_t, self.auc_r)
                if self.levene[1] <= self.alpha and self.auc_log==False:
                    self.log_auc()
                    self.levene = self.get_levene(self.auc_t, self.auc_r)
            else:
                self.f = self.get_f(self.auc_t, self.auc_r)
                self.levene = self.get_levene(self.auc_t, self.auc_r)
                if self.auc_log==False and (self.f[1] <= self.alpha or
                                           self.levene[1] <= self.alpha):
                    self.log_auc()
                    self.f = self.get_f(self.auc_t, self.auc_r)
                    self.levene = self.get_levene(self.auc_t, self.auc_r)
            self.anova = self.get_anova(self.auc_t, self.auc_r, self.auc)#0 - pd.DataFrame, 1 - bool
            self.oneside = self.get_oneside(self.auc_t, self.auc_r, self.anova[0])
        else:
            pass
