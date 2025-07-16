1. PRELIMINARY STEPS

	Preparation of surface water model with HydroAS: 
	
		- Create HydroAS model 
		- When HydroAS model is finished with everything, HydroAS V... -> Global Parameters -> Ein -/Ausgabedateien -> Ausgabe ASCII OPEN
		- Open HydroAS application, the exe file and 2dm-Datei öffnen -> Prüfungen -> Modell prüfen
		- Open HydroAS application, the exe file and 2dm-Datei öffnen 
		-> Simulation -> Simulation Starten (it creates hydro_as-2d-inp file that we make changes on.)
		- Create initial files folder and put hydro_as-2d-inp and MODFLOW initial conditions file hydro_as-2d-inp.ic
		- Protect initial condition files of HydroAS like wtiefe_0.dat, sources-in.dat in another folder to initialize something
		
		
	Preparation of groundwater model with MODELMUSE:
	
		- Create the model cells  
		- Model -> MODFLOW Layer Groups -> Delete all the layers except Upper Aquifer Layer to have one singular layer
		- Data -> Data Sets -> Hydrology -> Define hydrological and hydraulic parameters
		- Data -> Data Sets -> Layer Definition -> Define upper and lower elevation of the singular layer
		- Model -> MODFLOW Packages and Programs -> Solvers -> IMS -> Linear -> Override -> BICGSTAB
		- No other package will be opened. The code activates RIV package itself
		- Model -> MODFLOW Output Control -> Head -> Save in external file -> Binary (the other option does not work)
		- Model -> MODFLOW Output Control -> Head -> Print in listing file
		- Model -> MODFLOW Output Control -> Budget -> Compact budget
		- Model -> MODFLOW Output Control -> Budget -> Save cell flows -> Both (MF6)
		- Model -> MODFLOW Time -> Define starting and ending Time, max first time step length 
		- Model -> MODFLOW Time -> Define max first time step length and total time is divided by it, giving number of time steps
		- Model -> MODFLOW Time -> Steady State / Transient -> Transient
		- Model -> MODFLOW Options -> Wetting -> Use Newton formulation (MODFLOW-6)
		- Model -> MODFLOW Options -> Wetting -> Use Under_Relaxation option (MODFLOW-6)
		- File -> Export -> MODFLOW Input Files -> Save 
		- File -> Export -> MODFLOW Input Files -> Execute model (to see it runs without problem)
		
		
2. MAIN STEPS		

	Code of Preliminary Steps:
	
		- Open preliminary_files_code_5_last.py
		- Change the paths of files created on the lines
			- 194: nrows, ncols -> Changed to total number of rows and columns in your MODFLOW model
			- 410: volumen.dat file created by HydroAS 
			- 454: volumen_path
		- Except the paths defined, the only requirement is to have all HydroAS, MODFLOW, and code files in the same directory
		- Do not forget that MODFLOW nam file model_2_Riv_3.nam adds River Package to the file
		so if any problem occurs in preliminary step and you need to re-run this code, delete
		that line and re-run the code. The rest files are just created again. 
		- For uniform model, use uniform_preliminary_code.py code


		
		
	Main Code:
	
		- Open model_2_Riv_3.py
		- Change the paths of files and values on the lines
			- 26: hydro_as_path
			- 27: working_directory: folder of the main place with codes, input files, and models
			- 28: depth_file: From Data-out folder of HydroAS and the depth.dat file
			- 38: total_time (total simulation time)
			- 39: time_step (coupling frequency)
			- 114: nstp (number of time steps in MODFLOW defined)
			- 192: src_folder 
			- 193: dst_root
			- 314: conductance (river conductance value)
			- 360: MAXBOUND 10000 -> I put 10000 to make sure but to make sure, write 
			the number of total cells that River Package is applied. It can be done by
            counting these cells in model_2_Riv_3.riv file.			
			- 1160: workspace (define the folder that all the files are in together
			- 1437: depth_file_path (define the HydroAS Data-out -> AS -> depth.dat in your directory)
			- 1444: depth_file_path (same path with line 1437)
			- 1484: wtiefe_file (wtiefe_0.dat file in HydroAS Data-in -> AS)
			- 1486: output_wtiefe_file (same path with 1484)
			- 1540- 1542: You need to change UNIT X and TIME STEP Y based on what is
			defined as UNIT and nstp (TIME STEP), and have the same spacing in model_2_Riv_3.lst file 
			for these lines defined.
			- 1557: ncols (change to 
			your total columns in MODFLOW)
			- 1697: veloc_file (change to HydroAS Data-out -> AS -> veloc.dat 
			- 1698: nodeniederschlag_file (change to HydroAS Data-in -> AS -> nodeniederschlag.dat
			- 1699: geschw_0_file (change to HydroAS Data-in -> AS -> geschw_0.dat
			- 1834: input_file (change to HydroAS Data-in -> AS -> sources-in.dat
			- 1835: output_file (change to HydroAS Data-in -> AS -> sources-in.dat
			- 1881: geschw_0_file (change to coupling frequency defined)
			- 1883: results_dir_merge (put to Data-out folder)

END
