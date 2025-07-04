User Guide
===============

.. role:: raw-latex(raw)
   :format: latex
..

.. container:: flushleft

In AWSEM coarse grained simulations, the amino acids are represented by six particles, (CA, CB, O, C, N and H) except for Proline and Glycine both of which are represented by five particles. For Proline, no hydrogen is connected to the nitrogen inside its amide group. Glycine has no CB. Among those 6 particles in the standard representation, C, N and H are designated as "virtual sites" which means their coordinates are not dynamical variables but instead are computed based on the positions of the other particles, which are dynamical variables.

The standard AWSEM potential is made up of several term:

.. math::

   \begin{aligned}
   V_{AWSEM} = V_{con} + V_{chain} + V_{chi} + V_{rama} + V_{excl} + V_{contact} + V_{beta} + V_{pap} + V_{frag}
   \end{aligned}

Connectivity term
-----------------

The connectivity term is designed to maintain the bonded distances between :math:`C\alpha_i` and :math:`O_i`, :math:`C\beta_i` and :math:`C\alpha_{i+1}`. and between :math:`O_i` to :math:`C\alpha_{i+1}`.

.. math::

   \begin{aligned}
   V_{con} &= k_{con}(\sum_i^N (r_{C\alpha_i O_i} - r_{C\alpha O}^0)^2 +\sum_{res_i != GLY}   (r_{C\alpha_i C_{\beta_i}} - r_{C\alpha \beta}^0)^2 \\
   &+ \sum_i^{N-1} ((r_{C\alpha_i C\alpha_{i+1}} - r_{C\alpha C\alpha_{i+1} }^0)^2 + (r_{O_i C\alpha_{i+1}} - r_{O C\alpha_{i+1} }^0)^2))
   \end{aligned}

Chain term
----------

The chain term models the positions of C’ and N atoms.

.. math::

   \begin{aligned}
   V_{chain} = \lambda_{chain} [ \sum_{i=2}^N (\boldsymbol{r}_{N_i C_{\beta_i}} - \boldsymbol{r}^0_{N_i C_{\beta_i}})^2 +\sum_{i=1}^{N-1} (\boldsymbol{r}_{C'_i C_{\beta_i}} - \boldsymbol{r}^0_{C'_i C_{\beta_i}})^2 +\sum_{i=2}^{N-1} (\boldsymbol{r}_{N_i C'} - \boldsymbol{r}^0_{N C'})^2]
   \end{aligned}

We implemented the connectivity term and the chain term using "HarmonicBondForce".

Chirality term
--------------

The chirality term is used to fix the direction of the :math:`C_{\beta_i}` relative to the plane formed by :math:`C'_i`, :math:`C_{\alpha_i}` and :math:`N_i`.

.. math::

   \begin{aligned}
   V_{\chi} = \lambda_{\chi} \sum (\chi_i - \chi_0)^2 \\
   \chi_i = \frac{(\boldsymbol{r}_{C'_i C_{\alpha_i}} \times \boldsymbol{r}_{C_{\alpha_i N_i}}) }{|\boldsymbol{r}_{C'_i C_{\alpha_i}} \times \boldsymbol{r}_{C_{\alpha_i N_i}}| } \cdot \frac{\boldsymbol{r}_{C_{\alpha_i C_{\beta_i}}}}{|\boldsymbol{r}_{C_{\alpha_i C_{\beta_i}}}|}
   \end{aligned}

Rama term
---------

The rama term is used to fix the :math:`\phi`, :math:`\psi` angles within a reasonable range.

.. math::

   \begin{aligned}
   V_{rama} = - \lambda_{rama} \sum_{i=2}^{N-1} \sum_j W_j e^{-\sigma_j (\omega_{\phi_j} (\cos(\phi_i -\phi^0_j) -1 )^2 + \omega_{\psi_j} (\cos(\psi_i -\psi^0_j) -1 )^2 ) }
   \end{aligned}

The chirality term :math:`V_{\chi}` and Rama term was implemented using "CustomCompoundBondForce".

Excluded Volume term
--------------------

The excluded volume term prevents the overlapping of backbone atoms.

.. math::

   \begin{aligned}
   V_{excl} &= \lambda_{excl} \sum_{ij} [ H(r_{C_i C_j} - r^C_{ex})(r_{C_i C_j} - r^C_{ex})^2 + H(r_{O_i O_j} - r^O_{ex})(r_{O_i O_j} - r^O_{ex})^2] \\
   H(r) &= \begin{array}{cc}
   \{ &
   \begin{array}{cc}
   1 & x\geq 0 \\
   0 & x \le 0 \\
   \end{array}
   \end{array}
   \end{aligned}

The excluded volume term used "CustomNonbondedForce". All the parameters are the same as those defined in the original AWSEM paper. The parameters are defined in Table `1 <#tab:parameters>`__

.. container:: center

   .. container::
      :name: tab:parameters

      .. table:: parameters

         ===================================== ===== ======================
         Parameter                             Value Units
         ===================================== ===== ======================
         :math:`\lambda_{con}`                 120   kcal/:math:`Å^2` mol
         :math:`\lambda_{chain}`               120   kcal/:math:`Å^2` mol
         :math:`\lambda_{\chi}`                60    kcal/ mol
         :math:`\lambda_{rama}`                2     kcal/ mol
         :math:`\lambda_{excl}`                20    kcal/:math:`Å^2` mol
         :math:`r^0_{C\alpha_i C\alpha_{i+1}}` 3.816 :math:`Å`
         :math:`r^0_{C\alpha_i CO_i}`          2.40  :math:`Å`
         :math:`r^0_{CO_i C\alpha_i}`          2.76  :math:`Å`
         :math:`r^0_{C\alpha_i C\beta_i}`      1.53  :math:`Å`
         :math:`r^0_{N_i C\beta_i}`            2.46  :math:`Å`
         :math:`r^0_{C'_i C\beta_i}`           2.52  :math:`Å`
         :math:`r^0_{N_i C'_i}`                2.46  :math:`Å`
         :math:`\chi_0`                        -0.71 :math:`Å^3`
         ===================================== ===== ======================

.. container:: center

   .. container::
      :name: ramaParameters

      +---------------------+------------------------+-------------+------------+---------------+
      |                     | General Case           | Alpha Helix | Beta Sheet | Proline       |
      +=====================+========================+=============+============+===============+
      | W                   | 1.3149 1.32016 1.0264  | 2.0         | 2.0        | 2.17 2.15     |
      +---------------------+------------------------+-------------+------------+---------------+
      | :math:`\sigma`      | 15.398 49.0521 49.0954 | 419.0       | 15.398     | 105.52 109.09 |
      +---------------------+------------------------+-------------+------------+---------------+
      | :math:`\omega_\phi` | 0.15 0.25 0.65         | 1.0         | 1.0        | 1.0 1.0       |
      +---------------------+------------------------+-------------+------------+---------------+
      | :math:`\phi_0`      | -1.74 -1.265 1.041     | -0.895      | -2.25      | -1.153 -0.95  |
      +---------------------+------------------------+-------------+------------+---------------+
      | :math:`\omega_\psi` | 0.65 0.45 0.25         | 1.0         | 1.0        | 0.15 0.15     |
      +---------------------+------------------------+-------------+------------+---------------+
      | :math:`\psi_0`      | 2.138 -0.318 0.78      | -0.82       | 2.16       | 2.4 -0.218    |
      +---------------------+------------------------+-------------+------------+---------------+

Contact term
------------

The transferable interactions have the form:

.. math::

   \begin{aligned}
   V_{contact} &= V_{direct} + V_{water} \\
   V_{direct} &= \sum_{j-i>9} \gamma_{ij}(a_i, a_j) \Theta_{i,j}^{I} \\
   V_{water}(i,j) &= \sum_{j-i>9}\Theta_{i,j}^{II} (\sigma_{ij}^{wat} \gamma_{ij}^{wat}(a_i, a_j)  +  \sigma_{ij}^{prot} \gamma_{ij}^{ prot_{wat} }(a_i, a_j) ) \\
   \Theta_{i,j}^{\mu} &= \frac{1}{4} (1 + \tanh(\eta ( r_{ij} - r_{min}^{\mu})) ) (1 + \tanh(\eta ( r_{max}^{\mu} -r_{ij}  )) ) \\
   \sigma_{ij}^{water} &= \frac{1}{4} (1 - \tanh(\eta_{\sigma} ( \rho_i - \rho_0)) )  (1 - \tanh(\eta_{\sigma} ( \rho_j - \rho_0)) ) \\
   \sigma_{ij}^{prot} &= 1 - \sigma_{ij}^{water}
   \end{aligned}

:math:`\beta`-hydrogen bonding and P-AP terms
---------------------------------------------

We made some modification of these terms in order to make more efficient implementation of the force fields.

.. math::

   \begin{aligned}
   \theta_{i,j} = exp(-\frac{(r_{O_i N_j}-r_{ON})^2}{2\sigma_{ON}^2}-\frac{(r_{O_iH_j}-r_{OH})^2}{2 \sigma_{OH}^2}) \\
   \theta_{j,i} = exp(-\frac{(r_{O_j N_i}-r_{ON})^2}{2\sigma_{ON}^2}-\frac{(r_{O_jH_i}-r_{OH})^2}{2 \sigma_{OH}^2} )\\
   \theta_{j,i+2} = exp(-\frac{(r_{O_j N_{i+2}}-r_{ON})^2}{2\sigma_{ON}^2}-\frac{(r_{O_jH_{i+2}}-r_{OH})^2}{2 \sigma_{OH}^2}) \\
   V1_{ij} = \lambda_1(i,j)\theta_{i,j}\\
   V2_{ij} =\lambda_2(i,j)\theta_{i,j}\theta_{j,i}\\
   V3_{ij} = \lambda_3(i,j)\theta_{i,j}\theta_{j,i+2}  \\
   V_{ij} =V1_{ij} + V2_{ij} + V3_{ij}\\
   V_{beta} = -k_{beta} \sum_{ij} V_{ij}
   \end{aligned}

In previous the LAMMPS implementation, :math:`V_{beta} = -k_{beta} \sum_{ij} V_{ij} v_i v_j`, the additional term :math:`v_i v_j` was used to ensure that the hydrogen bonds do not occur within a span of 5 residues that is shorter than 12\ :math:`\AA`. Now this constraint is incorporated onto the pap term. The :math:`V_{beta}` defined here can be fit into the "CustomHbondForce" template. Since for :math:`V2_{ij}`, we can define :math:`O_i, N_i, H_i`, the oxygen, hydrogen and nitrogen of residue i as the donor, and :math:`N_j, H_j, O_j` as the acceptor. We could have implemented the exact same version as the LAMMPS version using "CustomCompoundBondForce", but computing bonded forces is much slower than computing non-bonded forces like "CustomHbondForce". When two residues are far apart, computing their interaction is unnecessary.

.. math::

   \begin{aligned}
   v_{i} = \frac{1}{2}(1+\tanh({\mu_1}*(r_{ca_{i} ca_{i+4}} -rc_{HB}))) \\
   \theta^1_{i,j} =  \frac{1}{2}(1+\tanh({\eta_{pap}}*(r_0 - r_{ca_{i} n_{j}}))) \\
   \theta^2_{i,j} =  \frac{1}{2}(1+\tanh({\eta_{pap}}*(r_0 - r_{ca_{i+4} n_{j+4}}))) \\
   \theta^3_{i,j} =  \frac{1}{2}(1+\tanh({\eta_{pap}}*(r_0 - r_{ca_{i+4} n_{j-4}}))) \\
   V_{i,j} = (\gamma_1(i, j)+\gamma_2(i,j) \theta^1_{i,j} \theta^2_{i,j} + \gamma_3{i,j} \theta^3_{i,j})v_i\\
   V_{pap} = \sum_{i,j} k_{pap} V_{i,j}
   \end{aligned}
