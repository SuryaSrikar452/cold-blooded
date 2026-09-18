# FINAL_REFERENCES.md: Academic & Methodological Bibliography

**Project:** SH-405 — Synthetic Patient Data Generation for Clinical Research  
**Target Model:** `GaussianCopulaFinal`  
**Date:** 2026-09-18  

---

## 1. Copula Theory & Latent Dependence

1. **Sklar, A. (1959).**  
   *Fonctions de répartition à n dimensions et leurs marges.*  
   Publications de l'Institut de Statistique de l'Université de Paris, 8, 229–231.  
   *(Foundational theorem establishing the decomposition of multivariate joint distribution functions into univariate marginal distributions and a copula function).*

2. **Nelsen, R. B. (2006).**  
   *An Introduction to Copulas.*  
   Springer Science & Business Media, 2nd Edition. New York: Springer.  
   *(Comprehensive mathematical reference on copula properties, parametric families, and Archimedean/elliptical copulas).*

3. **Genest, C., & Favre, A. C. (2007).**  
   *Everything you always wanted to know about copula modeling but were afraid to ask.*  
   Journal of Hydrologic Engineering, 12(4), 347–368.  
   *(Standard guide on inference functions for margins, probability integral transforms, and empirical rank copulas).*

---

## 2. Rank Inversion & Discrete Variable Correlation

4. **Kruskal, W. H. (1958).**  
   *Ordinal measures of association.*  
   Journal of the American Statistical Association, 53(284), 814–861.  
   *(Derivation of the exact mathematical relationship between Spearman's rank correlation $\rho_s$ and Gaussian copula correlation $R$: $R = 2 \sin(\frac{\pi}{6} \rho_s)$).*

5. **Pearson, K. (1909).**  
   *On a new method of determining correlation between a measured character A and a character B, of which only the percentage of cases wherein B exceeds (or falls short of) a given intensity is recorded for each grade of A.*  
   Biometrika, 7(1/2), 96–105.  
   *(Original derivation of the biserial correlation coefficient and the attenuation factor $\lambda = \phi(\tau) / \sqrt{p(1-p)}$ arising from latent dichotomization).*

6. **Olsson, U. (1979).**  
   *Maximum likelihood estimation of the polychoric correlation coefficient.*  
   Psychometrika, 44(4), 443–460.  
   *(Methodology for estimating underlying latent Gaussian correlation when continuous latent variables are categorized into discrete ordinal classes).*

7. **Higham, N. J. (2002).**  
   *Computing the nearest correlation matrix—a problem from finance.*  
   IMA Journal of Numerical Analysis, 22(3), 329–343.  
   *(Algorithm for projecting non-positive semidefinite correlation matrices onto the positive semidefinite cone using spectral decomposition and Dykstra's alternating projection).*

---

## 3. Two-Part & Zero-Inflated Hurdle Modeling

8. **Cragg, J. G. (1971).**  
   *Some statistical models for limited dependent variables with application to the demand for durable goods.*  
   Econometrica: Journal of the Econometric Society, 39(5), 829–844.  
   *(Introduction of the two-part hurdle model decoupling the zero point mass occurrence from positive continuous magnitude modeling).*

9. **Mullahy, J. (1986).**  
   *Specification and testing of some modified count data models.*  
   Journal of Econometrics, 33(3), 341–365.  
   *(Mathematical formulation of hurdle models for count and semi-continuous clinical variables with excessive zeros).*

---

## 4. Synthetic Patient Generation & Deep Generative Baselines

10. **Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K. (2019).**  
    *Modeling tabular data using conditional GAN.*  
    Advances in Neural Information Processing Systems (NeurIPS 2019), 32, 7335–7345.  
    *(Introduction of CTGAN and mode-specific normalization for mixed-type tabular synthesis).*

11. **Patki, N., Wedge, R., & Veeramachaneni, K. (2016).**  
    *The Synthetic Data Vault.*  
    IEEE International Conference on Data Science and Advanced Analytics (DSAA 2016), 399–410.  
    *(Original architecture for generative copula-based synthesis of relational databases and tabular data).*

12. **Choi, E., Biswal, S., Malin, B., Duke, J., Stewart, W. F., & Sun, J. (2017).**  
    *Generating multi-label discrete patient records using generative adversarial networks.*  
    Machine Learning for Healthcare Conference (PMLR 2017), 286–305.  
    *(Foundational medical synthetic data work on generative modeling for clinical cohorts).*

---

## 5. Evaluation Methodology & Model Reporting

13. **Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., ... & Gebru, T. (2019).**  
    *Model Cards for Model Reporting.*  
    Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT* 2019), 220–229.  
    *(Framework for transparent, standardized machine learning model reporting and ethical boundary documentation).*

14. **Hernandez, M., Epelde, G., Alberdi, A., Cilla, R., & Rankin, D. (2022).**  
    *Synthetic data in healthcare: A systematic review on metrics and validation techniques.*  
    Journal of Biomedical Informatics, 131, 104099.  
    *(Systematic taxonomy of fidelity, utility, and privacy metrics for validating synthetic clinical datasets).*

15. **Saito, T., & Rehmsmeier, M. (2015).**  
    *The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets.*  
    PLoS ONE, 10(3), e0118432.  
    *(Statistical justification for reporting PR-AUC alongside ROC-AUC when evaluating clinical minority classes such as diabetes).*
