
load "{BN}.xyz",XYZ
as sticks, XYZ
util.cbaw
load "{BN}_HOMO.cube",HOMO
load "{BN}_LUMO.cube",LUMO
load "{BN}_eplot.cube",eplot

cmd.volume_ramp_new('eldens', [0.02, 0.00, 0.00, 1.00, 0.00, 0.03, 0.00, 1.00, 1.00, 0.20, 0.06, 0.00, 0.00, 1.00, 0.00, ])
cmd.volume_ramp_new('eplot', [-0.04, 0.00, 0.00, 1.00, 0.12, -0.01, 0.00, 1.00, 1.00, 0.00, 0.14, 0.00, 1.00, 0.00, 0.02, 0.53, 1.00, 1.00, 0.00, 0.06, 1.00, 1.00, 0.50, 0.00, 0.01, 1.82, 1.00, 0.00, 0.00, 0.09, ])
cmd.volume_ramp_new('homo', [-0.005, 1.00, 0.00, 0.00, 0.050, 0.00, 0.96, 0.12, 0.80, 0.00, 0.00, 0.00, 0.98, 0.93, 0.00, 0.005, 0.00, 0.00, 1.00, 0.050, ])
cmd.volume_ramp_new('lumo', [-0.005, 1.00, 1.00, 0.00, 0.050, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.005, 0.00, 1.00, 1.00, 0.050, ])
as sticks,XYZ
util.cbaw
volume HOMO_volume, HOMO, homo
disable HOMO_volume
volume LUMO_volume, LUMO, lumo
disable LUMO_volume
volume eplot_volume, eplot, eplot
disable eplot_volume
isosurface HOMO_iso, HOMO, 0.005
set surface_color, tv_red, HOMO_iso
set surface_negative_color, tv_blue, HOMO_iso
set surface_negative_visible, on, HOMO_iso
disable HOMO_iso
isosurface LUMO_iso, LUMO, 0.005
set surface_color, cyan, LUMO_iso
set surface_negative_color, tv_orange, LUMO_iso
set surface_negative_visible, on, LUMO_iso
disable LUMO_iso
isosurface eplot_iso, eplot, 0.01
set surface_color, lime, eplot_iso
set surface_negative_color, ruby, eplot_iso
set surface_negative_visible, on, eplot_iso
disable eplot_iso
set transparency, 0.2