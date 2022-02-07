# -----------------------------------------------------------------
#  settings for dataset with 5 parameters
# -----------------------------------------------------------------
# Parameter names and their respective intervals
# 1. haloMassLog        interval=[8.0, 15.0]
# 2. redshift           interval=[6.0, 13.0]
# 3. sourceAge          interval=[0.1, 20.0]
# 4. qsoAlpha           interval=[1.0, 2.0]
# 5. starsEscFrac       interval=[0.0, 1.0]

p5_limits = [[8.0, 15.0],
             [6.0, 13.0],
             [0.1, 20.0],
             [1.0, 2.0],
             [0.0, 1.0],
             ]

p5_names_latex =['\log_{10}{\mathrm{M}_\mathrm{halo}}',
                 'z',
                 't_{\mathrm{source}}',
                 '-\\alpha_{\mathrm{QSO}}',
                 'f_{\mathrm{esc,\\ast}}'
                 ]


# -----------------------------------------------------------------
#  settings for dataset with 8 parameters
# -----------------------------------------------------------------
#        1. haloMassLog         interval=[8.0, 15.0]
#        2. redshift            interval=[6.0, 13.0]
#        3. sourceAge           interval=[0.1, 20.0]
#        4. qsoAlpha            interval=[0.0, 2.0]
#        5. qsoEfficiency       interval=[0.0, 1.0]
#        6. starsEscFrac        interval=[0.0, 1.0]
#        7. starsIMFSlope       interval=[0.0, 2.5]
#        8. starsIMFMassMinLog  interval=[0.6989700043360189, 2.6989700043360187]

p8_limits = [[8.0, 15.0],
             [6.0, 13.0],
             [0.1, 20.0],
             [0.0, 2.0],
             [0.0, 1.0],
             [0.0, 1.0],
             [0.0, 2.5],
             [0.6989700043360189, 2.6989700043360187]  # log_10 of [5, 500]
             ]

p8_names_latex = ['\log_{10}{\mathrm{M}_\mathrm{halo}}',
                  'z',
                  't_{\mathrm{source}}',
                  '-\\alpha_{\mathrm{QSO}}',
                  '\\epsilon_{\mathrm{QSO}}',
                  'f_{\mathrm{esc,\\ast}}',
                  '\\alpha_{\mathrm{IMF,\\ast}}',
                  '\log_{10}{\mathrm{M}_{\mathrm{min}, \\ast}}'
                 ]

