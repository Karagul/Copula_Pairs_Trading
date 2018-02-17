class quant_models(price_streaming):    
    def price_return(self,dataset,field,i):
        returns = np.log(dataset[self.instruments[i]+"_"+field]/dataset[self.instruments[i]+"_"+field].shift(1))
        return returns                                
    def copula_params(self,family,dataset):
        tau=kendalltau(x=dataset[self.instruments[0]+"_mid"],y=dataset[self.instruments[1]+"_mid"])[0]
        rho=np.sin(np.pi/2*tau)
        if  family == 'clayton':
            return 2*tau/(1-tau)
        elif family == 'frank':
            integrand = lambda t: t/(np.exp(t)-1)
            frank_fun = lambda theta: ((tau - 1)/4.0  - (quad(integrand, sys.float_info.epsilon, theta)[0]/theta - 1)/theta)**2
            return minimize(frank_fun, 4, method='BFGS', tol=1e-5).x
        elif family == 'gumbel':
            return 1/(1-tau)    
        elif family == 'student-t':            
            return rho
        
    def log_pdf_copula(self, family,dataset,student_df=None):
        theta=self.copula_params(family,dataset)
        returns_0=self.price_return(dataset,'mid',0) #x
        returns_1=self.price_return(dataset,'mid',1) #y
        u=ECDF(returns_0)(returns_0.tail(1))
        v=ECDF(returns_1)(returns_1.tail(1))
        if  family == 'clayton':
            pdf = (theta+1) * ((u**(-theta)+v**(-theta)-1)**(-2-1/theta)) * (u**(-theta-1)*v**(-theta-1))
        elif family == 'frank':
            num = -theta *(np.exp(-theta)-1) * (np.exp(-theta*(u+v)))
            denom = ((np.exp(-theta*u)-1) * (np.exp(-theta*v)-1) + (np.exp(-theta)-1))**2
            pdf = num/denom
        elif family == 'gumbel':
            A = (-np.log(u))**theta + (-np.log(v))**theta
            #C = np.exp(-A**(1/theta))
            C = np.exp(-((-np.log(u))**theta + (-np.log(v))**theta)**(1/theta))
            pdf = C * (u*v)**(-1) * (A**(-2+2/theta))*((np.log(u)*np.log(v))**(theta-1))*(1+(theta-1)*A**(-1/theta))
        elif family=='student-t':
            rho = theta
            n = student_df
            pdf_x  =  lambda x: gamma((n+1)/2)/(np.sqrt(n*np.pi)*gamma(n/2))*(1+x**2/n)**(-(n+1)/2) #pdf of x
            #pdf_xy =  lambda x,y: 1/(gamma(n/2))*gamma((n+2)/2)/(n*np.pi) *  (1/np.sqrt(rho))* (1 + (x**2)/(n*rho)+(y**2)/(n*rho))**(-((n+2)/2))     #joint pdf of x,y
            tn     =  lambda h: quad(pdf_x, -math.inf, h)[0]     #CDF of x
            #t2n   =  lambda h,k : dblquad (pdf_xy, -math.inf,h, lambda x: -math.inf,lambda x:k)[0]        #CDF of x,y          
            pdf    =   1/np.sqrt(1-rho**2) * gamma((n+2)/2)*gamma((n/2))/(gamma((n+1)/2)**2) * ((1+inv(tn)(u)**2/n)*(1+inv(tn)(v)**2/n))**((n+1)/2) / (1+1/(n*(1-rho**2))*(inv(tn)(u)**2-2*rho*inv(tn)(u)*inv(tn)(v)+inv(tn)(v)**2))**((n+2)/2)            
        return np.log(pdf)
###################################################################
#Alternative for optimizing degree of freedom for student t
#    def opt_df(self,dataset,n):
#        AIC_values={'student-t':[]}
#        which = lambda lst:list(np.where(lst))[0][0]
#        for i in range(1,n):
#            log_pdf = self.log_pdf_copula(family='student-t',dataset=dataset,student_df=i)
#            loglikehood=sum(np.nan_to_num(log_pdf))
#            AIC_values['student-t'].append((-2*loglikehood+2))
#        AIC_min    =  min(AIC_values['student-t'])
#        ls = list(map(lambda x:x==AIC_min, AIC_values['student-t']))
#        student_df=which(ls)
#        return student_df
###################################################################

    def opt_df(self,dataset):
        func = lambda student_df: -2*sum(np.nan_to_num(self.log_pdf_copula(family='student-t',dataset=dataset,student_df=student_df)))+2       
        AIC_min = round(minimize(func,x0=1,method='Nelder-Mead').x[0])
        #AIC_min = round(basinhopping(func,x0=20,stepsize=1).x[0])
        student_df = AIC_min
        return student_df 
        
        
###################################################################       
#Alternative for optimizing/chooseing the family of the coppula       
#    def opt_aic(self,dataset):
#        family=['clayton','frank','gumbel','student-t']
#        AIC_values={'clayton':[],'frank':[], 'gumbel':[],'student-t':[]}
#        for i in family:
#            if i == 'student-t':
#                student_df=self.opt_df(dataset)
#                log_pdf = self.log_pdf_copula(family=i,dataset=dataset,student_df=student_df)
#                loglikehood=sum(np.nan_to_num(log_pdf))
#                AIC_values[i]=-2*loglikehood+2 
#            else:
#                log_pdf = self.log_pdf_copula(family=i,dataset=dataset)
#                loglikehood=sum(np.nan_to_num(log_pdf))
#                AIC_values[i]=-2*loglikehood+2
#        AIC_values=pd.DataFrame.from_dict(AIC_values,orient='index')
#        AIC_min=AIC_values.min(axis=0)[0]
#        copula_type=AIC_values[AIC_values==AIC_min].dropna().index[0]
#        return copula_type, student_df
###############################################################################



    def opt_aic(self,dataset):
        family=['clayton','frank','gumbel','student-t']
        AIC_values={'clayton':[],'frank':[], 'gumbel':[],'student-t':[]}
        for i in family:
            if i == 'student-t':
                student_df=self.opt_df(dataset)
                log_pdf = self.log_pdf_copula(family=i,dataset=dataset,student_df=student_df)
                loglikehood=sum(np.nan_to_num(log_pdf))
                AIC_values[i]=-2*loglikehood+2 
            else:
                log_pdf = self.log_pdf_copula(family=i,dataset=dataset)
                loglikehood=sum(np.nan_to_num(log_pdf))
                AIC_values[i]=-2*loglikehood+2
        AIC_values=pd.DataFrame.from_dict(AIC_values,orient='index')
        AIC_min=AIC_values.min(axis=0)[0]
        copula_type=AIC_values[AIC_values==AIC_min].dropna().index[0]
        return copula_type, student_df
    
    def Misprice_Index(self,dataset):  
        family, student=self.opt_aic(dataset)      
        theta=self.copula_params(family,dataset)
        returns_0=self.price_return(dataset,'mid',0) #x
        returns_1=self.price_return(dataset,'mid',1) #y
        u=ECDF(returns_0)(returns_0.tail(1))
        v=ECDF(returns_1)(returns_1.tail(1))
        MI_0=None
        MI_1=None
        if  family == 'clayton':
            MI_0=v**(-theta-1)*(u**(-theta)+v**(-theta)-1)**(-(1/theta)-1)
            MI_1=u**(-theta-1)*(u**(-theta)+v**(-theta)-1)**(-(1/theta)-1)
        elif family == 'frank':
            MI_0=((np.exp(-theta*u)-1)*(np.exp(-theta*v)-1)+(np.exp(-theta*u)-1))/  \
                 ((np.exp(-theta*u)-1)*(np.exp(-theta*v)-1)+(np.exp(-theta)-1))
            MI_1=((np.exp(-theta*u)-1)*(np.exp(-theta*v)-1)+(np.exp(-theta*v)-1))/  \
                 ((np.exp(-theta*u)-1)*(np.exp(-theta*v)-1)+(np.exp(-theta)-1))  
        elif family == 'gumbel':
            A  = (-np.log(u))**theta +(-np.log(v))**theta
            #C = np.exp(-A**(1/theta)) #Gumbel copula
            C    = np.exp(-((-np.log(u))**theta + (-np.log(v))**theta)**(1/theta))
            MI_0 = C*(A**((1-theta)/theta))*(-np.log(v))**(theta-1)*(1/v)
            MI_1 = C*(A**((1-theta)/theta))*(-np.log(u))**(theta-1)*(1/u)
        elif family== 'student-t':
            rho=theta
            n = student_df
            pdf_x     = lambda x: gamma((n+1)/2)/(np.sqrt(n*np.pi)*gamma(n/2))*(1+x**2/n)**(-(n+1)/2) #pdf of x
            pdf_x_2   = lambda x: gamma((n+1+1)/2)/(np.sqrt((n+1)*np.pi)*gamma((n+1)/2))*(1+x**2/(n+1))**(-(n+1+1)/2) #pdf of x with degree n+1
           # pdf_xy    = lambda x,y: 1/(gamma(n/2))*gamma((n+2)/2)/(n*np.pi) *  (1/np.sqrt(rho))* (1 + (x**2)/(n*rho)+(y**2)/(n*rho))**(-((n+2)/2))     #joint pdf of x,y
            tn        = lambda h: quad(pdf_x, -math.inf, h)[0]     #CDF of x
            tn_2      = lambda h: quad(pdf_x_2, -math.inf, h)[0]     #CDF of x with degree n+1
            #t2n       = lambda h,k : dblquad (pdf_xy, -math.inf,h, lambda x: -math.inf,lambda x:k)[0] #CDF of x,y         
            MI_0      = tn_2(np.sqrt((n+1)/(n+inv(tn)(v)**2))*(inv(tn)(u)-rho*inv(tn)(v))/np.sqrt(1-rho**2))
            MI_1      = tn_2(np.sqrt((n+1)/(n+inv(tn)(u)**2))*(inv(tn)(v)-rho*inv(tn)(u))/np.sqrt(1-rho**2))
        print('family',family)
        return MI_0, MI_1
