# change similarity loss
# python main_cascaded.py --loss=NCC --reg=tv --alpha=2.0 --res_level=3
python main_cascaded.py --loss=MSE --reg=tv --alpha=2.0 --res_level=3


# # change resolution level
# SIM=""
# python main_cascaded.py --loss=$SIM --reg=tv --alpha=2.0 --res_level=1

# # change regularizer (using jacobian determinant)
# JACALPHA="" # 값 scale 확인 후 적용 필요
# python main_cascaded.py --loss=$SIM --reg=jac --alpha=$JACALPHA --res_level=1
# python main_cascaded.py --loss=$SIM --reg=jac --alpha=$JACALPHA --res_level=3

# # scaled tv hyperparam
# ALPHA=""
# SCA=""
# python main_cascaded.py --loss=$SIM --reg=tv --alpha=$ALPHA --sca_alp=$SCA --res_level=3