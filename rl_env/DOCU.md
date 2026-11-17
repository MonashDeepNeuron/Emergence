worth to mention:

1. layout. 
    a. the pad size must be a multiple of the block size (0.2m), otherwise round() builds a grid whose cells don't line up with the pad.  
    b. current code handles only rectangl shape single target pad.  
  
2. ai  
    a. part from claude and chatGPT:  
        **_check_dropable()**, **_choose_drop_cell**, **_empty_cells_in_view** functions and **reset** method in env.py .   
        **_sync()** method in rover.py.  
  
3. limitations  
    a. the limitations mentioned in layout.  
    b. changong the layout means the eng.py code also need to change. 
    c. *to be added*. 

