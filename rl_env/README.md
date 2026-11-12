some declarations

ai: _check_dropable(), _choose_drop_cell, _empty_cells_in_view functions and reset method in env.py

and _sync() method in rover.py are from claude

the _check_dropable(), _choose_drop_cell, _empty_cells_in_view are created to fix the bug of when dropping the last peace of material the robot will kept rotating.

the approach for fixing is:

instead of stop rewarding/penallising then entered the material zone and target pad, we stop rewarding as long as the robot entered the rctangle with the length material zone length + the robot length. like give it a margin where the robt will not get penalised

the drop point: 

change DROP_DIST = REACHABLE_LEN
 = 0.55,it's the maximum it could get not fixed. it's like reachable lenth, this is more like the drapable length, so i changed the constant name to DROPABLE_LEN


 and claude fixed the  logic part according to this