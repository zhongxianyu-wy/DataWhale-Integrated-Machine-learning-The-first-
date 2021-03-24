# task4
## 超参数调优
### pipeline管道的原理和使用
在使用超参数调优之前先了解下学习内容中使用的pipeline    
1）sklearn库的pipeline能够构建一套流水线，完成一连串数据挖掘的步骤，其特点是经过几步转化器后最后一步时模型评估器。   
2）其中每一步输出的结果是下一步的输入结果，常常包括数据的标准化、缺失值插补、降维处理等    
3）每一步用元组来表示即（'步骤的名称',具体步骤的函数）   
如：

	pipe = Pipeline([('sc',StandardScaler()),
                 ('pca',PCA(n_components=2)),
                 ('clf',LogisticRegression(random_state=20210324))   
                 ])
	pip.fit(trainX,trainY)
上述步骤则是对原始训练数据分别进行了标准化、PCA降维、逻辑斯蒂回归处理   
### 参数与超参数的理解   
参数：参数w是在我们确定了超参数λ后，通过确定损失函数然后经过梯度下降法等优化算法优化出来的     
超参数：则是无法优化出来，只能通过不同的尝试找到最后甚至是局部最优的参数    
## 方法1 网格搜索GridSearchCV()：
原理：但我们有k个超参数需要优化时，我们首先确定K个超参数有多少个预选值，根据k个维度的所有预选值进行排列组合，构成了一个K维的网格，而我们通过遍历这些网点，找到最优的那个点作为最佳超参数，但是依旧只能说给出网点的局部最优超参数    
## 方法2：随机搜索RandomizedSearchCV()：
与网格搜索的对比：网格搜索在预选值比较少稀疏时，可能达不到预期的情况，而较多时又会增加计算负担    
因此加入了对分布随机的思想，优化了维度灾难产生的算力问题以及更加有效地选取最优超参数   
##  示例 
### 网格搜索
	pipe_SVR = make_pipeline(("normalize",StandardScaler()),
                          ("SVR",SVR())	
	param_range = [0.0001,0.002,0.5,1.0,10.0,100.0]
	param_grid = [{"svr__C":param_range,"svr__kernel":["linear"]},  {"svr__C":param_range,"svr__gamma":param_range,"svr__kernel":["rbf"]}]
	gs = GridSearchCV(estimator=pipe_svr,
                  param_grid = param_grid,
                  scoring = 'r2',
                  cv = 10)    
	gs = gs.fit(X,y)

### 随机搜索  
	pipe_SVR = make_pipeline(("normalize",StandardScaler()),
                          ("SVR",SVR())	
	distributions = dict(svr__C=uniform(loc=1.0, scale=4), 
                     svr__kernel=["linear","rbf"],                                   
                     svr__gamma=uniform(loc=0, scale=4))

	rs = RandomizedSearchCV(estimator=pipe_svr,
                        param_distributions = distributions,
                        scoring = 'r2',
                        cv = 10)     
	rs = rs.fit(X,y)
不同的时参数设置那块加入了分布设定随机函数   

## 方法3：贝叶斯调参
	pipe_SVR = make_pipeline(("normalize",StandardScaler()),
                          ("SVR",SVR(svr__C=svr__C,svr__kernel=svr__kernel,svr__gamma=svr__gamma))
	bayes_svr = BayesianOptimization(
    pipe_SVR, 
    {
     svr__C:uniform(loc=1.0, scale=4), 
     svr__kernel:["linear","rbf"],                                   
     svr__gamma:uniform(loc=0, scale=4)
    }
	)
	bayes_svr.maximize(n_iter=10)
## 方法4：贪心算法  
原理：先使用当前对模型影响最大的参数进行调优，达到当前参数下的模型最优化，再使用对模型影响次之的参数进行调优，直到所有的参数调整完毕  
缺点：对比网格搜索，很可能局限于局部最优的情况   无法全局总览  