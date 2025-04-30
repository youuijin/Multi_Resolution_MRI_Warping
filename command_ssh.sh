# Deterministic
## Single resolution
python main_MrRegNet.py --out_layers=1 --batch_size=4 --loss=MSE --reg=tv --alpha=4.0 --alp_sca=1.0
## Multi resolution
python main_MrRegNet.py --out_layers=3 --batch_size=4 --loss=MSE --reg=tv --alpha=16.0 --alp_sca=0.5

# Probabilistic
## Single resolution
python main_ssh_uncert.py --out_layers=1 --batch_size=4 --loss=NLL --reg=KL --alpha=0.01 --alp_sca=1.0
python main_ssh_uncert.py --out_layers=1 --batch_size=4 --loss=NLL --reg=KL --alpha=0.001 --alp_sca=1.0
python main_ssh_uncert.py --out_layers=1 --batch_size=4 --loss=NLL --reg=KL --alpha=0.0001 --alp_sca=1.0
python main_ssh_uncert.py --out_layers=1 --batch_size=4 --loss=NLL --reg=atv --alpha=2.0 --alp_sca=1.0
python main_ssh_uncert.py --out_layers=1 --batch_size=4 --loss=NLL --reg=atv --alpha=4.0 --alp_sca=1.0
python main_ssh_uncert.py --out_layers=1 --batch_size=4 --loss=NLL --reg=atv_KL --alpha=2.0_0.001 --alp_sca=1.0
python main_ssh_uncert.py --out_layers=1 --batch_size=4 --loss=NLL --reg=atv_KL --alpha=4.0_0.001 --alp_sca=1.0
## Multi resolution
python main_ssh_uncert.py --out_layers=3 --batch_size=4 --loss=NLL --reg=KL --alpha=0.01 --alp_sca=1.0
python main_ssh_uncert.py --out_layers=3 --batch_size=4 --loss=NLL --reg=KL --alpha=0.001 --alp_sca=1.0
python main_ssh_uncert.py --out_layers=3 --batch_size=4 --loss=NLL --reg=KL --alpha=0.0001 --alp_sca=1.0
python main_ssh_uncert.py --out_layers=3 --batch_size=4 --loss=NLL --reg=atv --alpha=2.0 --alp_sca=1.0
python main_ssh_uncert.py --out_layers=3 --batch_size=4 --loss=NLL --reg=atv --alpha=4.0 --alp_sca=1.0
python main_ssh_uncert.py --out_layers=3 --batch_size=4 --loss=NLL --reg=atv_KL --alpha=2.0_0.001 --alp_sca=1.0
python main_ssh_uncert.py --out_layers=3 --batch_size=4 --loss=NLL --reg=atv_KL --alpha=4.0_0.001 --alp_sca=1.0