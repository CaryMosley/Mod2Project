def make_models(data, target, num_iter=5, models=['ols', 'lasso', 'ridge','enet'], complexity='simple', degree=3):
    '''This function takes in the features, target, model, and complexity to return
    r^2 value'''
    x_axis = np.arange(num_iter)
    ols_test = []
    lasso_test = []
    ridge_test = []
    enet_test = []
    cv = [10,20]
    sample_models = {}
    parameters = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]
    model_metrics = {}
    for val in cv:
        for i in range(num_iter):
            if complexity == 'simple':
                X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.2)
            elif complexity == 'polynomial':
                poly = PolynomialFeatures(degree=degree)
                Xp = poly.fit_transform(data)
                X_train, X_test, Y_train, Y_test = train_test_split(Xp, target, test_size=0.2)
            elif complexity == 'poly_intx':
                poly_intx = PolynomialFeatures(degree=degree, interaction_only=True)
                Xpn = poly_intx.fit_transform(data)
                X_train, X_test, Y_train, Y_test = train_test_split(Xpn, target, test_size=0.2)
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train=scaler.transform(X_train)
            X_test=scaler.transform(X_test)
            if 'ols' in models:
                linreg = LinearRegression()
                ols_model_tr = linreg.fit(X_train, Y_train)
                train_pred = ols_model_tr.predict(X_train)
                test_pred = ols_model_tr.predict(X_test)
                sample_models['ols'] = ols_model_tr
                test_score = ols_model_tr.score(X_test, Y_test)
                ols_test.append(test_score)
                model_metrics['ols_train_rmse'] = np.sqrt(mean_squared_error(Y_train, train_pred))
                model_metrics['ols_test_rmse'] = np.sqrt(mean_squared_error(Y_test, test_pred))
                model_metrics['ols_test_r2'] = ols_model_tr.score(X_test, Y_test)
                model_metrics['ols_train_r2'] = ols_model_tr.score(X_train, Y_train)
            if 'lasso' in models:
                #lasso_regressor = GridSearchCV(Lasso(), parameters, scoring='neg_mean_squared_error', cv = 5)
                #lasso_regressor.fit(X_train, Y_train)
                #alpha = list(lasso_regressor.best_params_.values())[0]
                lasso = LassoCV(alphas=parameters, cv = val)
                lasso_model_tr = lasso.fit(X_train, Y_train)
                train_pred = lasso_model_tr.predict(X_train)
                test_pred = lasso_model_tr.predict(X_test)
                sample_models['lasso'] = lasso_model_tr
                test_score = lasso_model_tr.score(X_test, Y_test)
                lasso_test.append(test_score)
                model_metrics['lasso_train_rmse'] = np.sqrt(mean_squared_error(Y_train, train_pred))
                model_metrics['lasso_test_rmse'] = np.sqrt(mean_squared_error(Y_test, test_pred))
                model_metrics['lasso_test_r2'] = lasso_model_tr.score(X_test, Y_test)
                model_metrics['lasso_train_r2'] = lasso_model_tr.score(X_train, Y_train)
            if 'ridge' in models:
                #ridge_regressor = GridSearchCV(Ridge(), parameters, scoring='neg_mean_squared_error', cv = 5)
                #ridge_regressor.fit(X_train, Y_train)
                #alpha = list(ridge_regressor.best_params_.values())[0]
                ridge = RidgeCV(alphas=parameters,cv = val)
                ridge_model_tr = ridge.fit(X_train, Y_train)
                train_pred = ridge_model_tr.predict(X_train)
                test_pred = ridge_model_tr.predict(X_test)
                sample_models['ridge'] = ridge_model_tr
                test_score = ridge_model_tr.score(X_test, Y_test)
                ridge_test.append(test_score)
                model_metrics['ridge_train_rmse'] = np.sqrt(mean_squared_error(Y_train, train_pred))
                model_metrics['ridge_test_rmse'] = np.sqrt(mean_squared_error(Y_test, test_pred))
                model_metrics['ridge_test_r2'] = ridge_model_tr.score(X_test, Y_test)
                model_metrics['ridge_train_r2'] = ridge_model_tr.score(X_train, Y_train)
            if 'enet' in models:
                enet = ElasticNetCV(alphas=parameters, l1_ratio = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],cv = val)
                enet_model_tr = enet.fit(X_train, Y_train)
                train_pred = enet_model_tr.predict(X_train)
                test_pred = enet_model_tr.predict(X_test)
                sample_models['enet'] = enet_model_tr
                test_score = enet_model_tr.score(X_test, Y_test)
                enet_test.append(test_score)
                model_metrics['enet_train_rmse'] = np.sqrt(mean_squared_error(Y_train, train_pred))
                model_metrics['enet_test_rmse'] = np.sqrt(mean_squared_error(Y_test, test_pred))
                model_metrics['enet_test_r2'] = enet_model_tr.score(X_test, Y_test)
                model_metrics['enet_train_r2'] = enet_model_tr.score(X_train, Y_train)
            i += 1
    if 'ols' in models:
        plt.plot(ols_test, label='ols')
    if 'ridge' in models:
        plt.plot(ridge_test, label='ridge')
    if 'lasso' in models:
        plt.plot(lasso_test, label='lasso')
    if 'enet' in models:
        plt.plot(enet_test, label='enet')
    plt.ylabel('R2 test score')
    plt.xlabel('number of iterations')
    plt.ylim((0.40, 0.99))
    plt.legend()
    print (lasso_model_tr.alpha)
    return sample_models, model_metrics