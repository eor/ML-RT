# -----------------------------------------------------------------
#  settings for dataset with 5 parameters
# -----------------------------------------------------------------
# Parameter names and their respective intervals
# 1. haloMassLog        interval=[8.0, 15.0]
# 2. redshift           interval=[6.0, 13.0]
# 3. sourceAge          interval=[0.1, 20.0]
# 4. qsoAlpha           interval=[1.0, 2.0]
# 5. starsEscFrac       interval=[0.0, 1.0]


p5_limits = list()
p5_limits.append([8.0, 15.0])
p5_limits.append([6.0, 13.0])
p5_limits.append([0.1, 20.0])
p5_limits.append([1.0, 2.0])
p5_limits.append([0.0, 1.0])


p5_names_latex = list()
p5_names_latex.append('\log_{10}{\mathrm{M}_\mathrm{halo}}')
p5_names_latex.append('z')
p5_names_latex.append('t_{\mathrm{source}}')
p5_names_latex.append('-\\alpha_{\mathrm{QSO}}')
p5_names_latex.append('f_{\mathrm{esc,\\ast}}')





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

p8_limits = list()
p8_limits.append([8.0, 15.0])
p8_limits.append([6.0, 13.0])
p8_limits.append([0.1, 20.0])
p8_limits.append([0.0, 2.0])
p8_limits.append([0.0, 1.0])
p8_limits.append([0.0, 1.0])
p8_limits.append([0.0, 2.5])
p8_limits.append([0.6989700043360189, 2.6989700043360187])  # 5 -> 500

p8_names_latex = list()
p8_names_latex.append('\log_{10}{\mathrm{M}_\mathrm{halo}}')
p8_names_latex.append('z')
p8_names_latex.append('t_{\mathrm{source}}')
p8_names_latex.append('-\\alpha_{\mathrm{QSO}}')
p8_names_latex.append('\\epsilon_{\mathrm{QSO}}')
p8_names_latex.append('f_{\mathrm{esc,\\ast}}')
p8_names_latex.append('\\alpha_{\mathrm{IMF,\\ast}}')
p8_names_latex.append('\log_{10}{\mathrm{M}_{\mathrm{min}, \\ast}}')


