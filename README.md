# Stiffening of a Corrugated Wing by a Regular Longitudinal Folding Pattern
This repository contains the MatLab code to derive Eqn. 2 in [L. Girardi, et al., IEEE IROS 
L. Girardi, R. Wu, Y. Fukatsu, H. Shigemune, S. Mintchev, “Biodegradable Gliding Paper Flyers Fabricated Through Inkjet Printing” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst, 2024] (not yet available online) and validate the stiffening ratio equation with finite element analysis. 

- The live MatLab script [_symbolic_second_area_moment.mlx_](https://github.com/erl-ethz/wing_folding_rigidity/blob/9155de9779e300c6a98384e2e15d242dfda4ee75/symbolic_second_area_moment.mlx) contains the symbolic derivation and the value substitution to compute the example in the paper body. It requires the symbolic math toolbox.
- The Python script [_creased_wing_bending.py_](https://github.com/erl-ethz/wing_folding_rigidity/blob/9155de9779e300c6a98384e2e15d242dfda4ee75/creased_wing_bending.py) uses Abaqus APIs to generate the folded wing pattern and the equivalent flat wing with the same unfolded geometry and constitutive properties, and computes the stiffening ratio. It runs on a system with Abaqus installed. Developed and tested in Abaqus 2024, backward compatibility is plausible, but not guaranteed.

If you use the data contained in this repository, please cite:
(to be added)
