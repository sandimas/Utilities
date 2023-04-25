using CSV
using DataFrames



########################################
##
## function load_from_SmoQyDQMC(;
##     simulationfolder::String, # Root folder for your simulation
##     correlation::String,      # Measurement, e.g. "greens_up", "pair"
##     space::String,            # "momentum" or "position"
##     type::String,             # "equal-time", "time-displaced", etc
##     bin::Bool=false)          # Using binned data or stats data
##                               #   e.g. files that end in "*bins.csv" vs "*stats.csv"
##  
## Data combinations not reported in csv will be filled with NaN values
##    e.g. if orbital combo 1,2 doesn't exist, you will get NaNs
## 
## For bin==false returns:
##   dimension_key: dictionary which maps stats.csv csv headers to dimensions in the three returned arrays
##                  e.g. Dict("ORBITAL_ID_2" => 2, "K_3" => 6, ...)
##   MEAN_R       : Real part of the data. 6 dimensional array with dimensions in order of
##                  ["___ID_1", "____ID_2", "TAU", "K/R_1", "K/R_2", "K/R_3"]
##                  For equal-time, length("TAU") = 1, For 1D/2D unused dimensions have length = 1
##   MEAN_I       : Imaginary part of the data. 6 dimensional array with same dimensions as MEAN_R
##   STD          : Standard deviation of data. 6 dimensional array with same dimensions as MEAN_R
##
## For bin==true returns
##   dimension_key: dictionary which maps stats.csv csv headers to dimensions in the three returned arrays
##                  e.g. Dict("ORBITAL_ID_2" => 2, "K_3" => 6, ...)
##   MEAN_R       : Real part of the data. 8 dimensional array with dimensions in order of
##                  ["___ID_1", "____ID_2", "TAU", "K/R_1", "K/R_2", "K/R_3", "BIN", "PID"]
##                  For equal-time, length("TAU") = 1, For 1D/2D unused dimensions have length = 1
##   MEAN_I       : Imaginary part of the data. 8 dimensional array with same dimensions as MEAN_R
##   SIGN_R       : Average sign for real part of data for that bin/PID
##   SIGN_I       : Average sign for real part of data for that bin/PID
##
## Test block and examples
#
# dict,real,imag,std = load_from_SmoQyDQMC(simulationfolder="/home/james/Documents/code/1D-RIXS/IO_test-1",
#                                          correlation="greens_up",
#                                          space="momentum",
#                                          type="time_displaced",bin=false)
#
# dict,real,image,sgnr,sgni = load_from_SmoQyDQMC(simulationfolder="/home/james/Documents/code/1D-RIXS/IO_test-1",
#                                                 correlation="greens_up",
#                                                 space="momentum",
#                                                 type="time_displaced",bin=false)


function load_from_SmoQyDQMC(;simulationfolder::String,
                              correlation::String,
                              space::String,
                              type::String,
                              bin::Bool=false)
    # Pre-format to fix likely typos                          
    _space = lowercase(space)
    _correlation = replace(lowercase(correlation),"-"=>"_")
    _type = replace(lowercase(type),"_"=>"-")

    # Load from appropriate data file type
    if bin
        file_end = "_bins.csv"
    else
        file_end = "_stats.csv"
    end
    csv_folder = simulationfolder * "/" * _type * "/" * _correlation * "/"
    csv_file = csv_folder * _correlation * "_" * _space * "_" * _type * file_end
    data_frame = CSV.read(csv_file,DataFrame)

    dim_prefix = "R_"
    if _space == "momentum"
        dim_prefix = "K_"
    end

    
    if bin == false

        dimensions = get_dimensions(df=data_frame,space=_space)
        space_size = ones(Int64,3)
        for dim in 1:dimensions
            space_size[dim] = maximum(data_frame[:,dim_prefix*string(dim)])+1
        end
        if (_correlation == "pair")
            ID1 = "BOND_ID_1"
            ID2 = "BOND_ID_2"
        else
            ID1 = "ORBITAL_ID_1"
            ID2 = "ORBITAL_ID_2"  
        end
        n_orbital = max(maximum(data_frame[:,ID1]),maximum(data_frame[:,ID2]))
        n_τ = (_type == "time-displaced") ?  maximum(data_frame[:,"TAU"])+1 : 1   
        dimension_key = Dict{String,Int}(
            ID1 => 1,
            ID2 => 2,
            "TAU" => 3,
            dim_prefix * "1" => 4,
            dim_prefix * "2" => 5,
            dim_prefix * "3" => 6
        )
        MEAN_R = fill!(Array{Float64}(undef,(n_orbital,n_orbital,n_τ,space_size[1],space_size[2],space_size[3])),NaN)
        MEAN_I = fill!(Array{Float64}(undef,(n_orbital,n_orbital,n_τ,space_size[1],space_size[2],space_size[3])),NaN)
        STD = fill!(Array{Float64}(undef,(n_orbital,n_orbital,n_τ,space_size[1],space_size[2],space_size[3])),NaN)

        for row in 1:length(data_frame[:,"MEAN_R"])
            o1 = data_frame[row,ID1]
            o2 = data_frame[row,ID2]
            τ = (n_τ != 1) ? data_frame[row,"TAU"]+1 : 1
            s1 = data_frame[row,dim_prefix*"1"]+1
            dimensions > 1 ? s2 = data_frame[row,dim_prefix*"2"]+1 : s2 = 1
            dimensions > 2 ? s3 = data_frame[row,dim_prefix*"3"]+1 : s3 = 1
            MEAN_R[o1,o2,τ,s1,s2,s3] = data_frame[row,"MEAN_R"] 
            MEAN_I[o1,o2,τ,s1,s2,s3] = data_frame[row,"MEAN_I"]
            STD[o1,o2,τ,s1,s2,s3] = data_frame[row,"STD"]  
        end
        return dimension_key, MEAN_R, MEAN_I, STD 
    else # Binned data
        
        index_file = csv_folder * _correlation * "_" * _space * "_" * _type * "_index_key.csv"
        index_frame = CSV.read(index_file,DataFrame)

            dimensions = get_dimensions(df=index_frame,space=_space)
            space_size = ones(Int64,3)
            for dim in 1:dimensions
                space_size[dim] = maximum(index_frame[:,dim_prefix*string(dim)])+1
            end
        if (_correlation == "pair")
            ID1 = "BOND_ID_1"
            ID2 = "BOND_ID_2"
        else
            ID1 = "ORBITAL_ID_ID_1"
            ID2 = "ORBITAL_ID_ID_2"  
        end
        n_orbital = max(maximum(index_frame[:,ID1]),maximum(index_frame[:,ID2]))
        n_τ = (_type == "time-displaced") ?  maximum(index_frame[:,"Tau"])+1 : 1   
        n_bin = maximum(data_frame[:,"BIN"])
        n_PID = maximum(data_frame[:,"BIN"]) + 1
        dimension_key = Dict{String,Int}(
            ID1 => 1,
            ID2 => 2,
            "TAU" => 3,
            dim_prefix * "1" => 4,
            dim_prefix * "2" => 5,
            dim_prefix * "3" => 6,
            "BIN" => 7,
            "PID" => 8
        )
        MEAN_R = fill!(Array{Float32}(undef,(n_orbital,n_orbital,n_τ,space_size[1],space_size[2],space_size[3],n_bin,n_PID)),NaN)
        MEAN_I = fill!(Array{Float32}(undef,(n_orbital,n_orbital,n_τ,space_size[1],space_size[2],space_size[3],n_bin,n_PID)),NaN)
        SIGN_R = fill!(Array{Float16}(undef,(n_orbital,n_orbital,n_τ,space_size[1],space_size[2],space_size[3],n_bin,n_PID)),NaN)
        SIGN_I = fill!(Array{Float16}(undef,(n_orbital,n_orbital,n_τ,space_size[1],space_size[2],space_size[3],n_bin,n_PID)),NaN)
        DATA_R = uppercase(_correlation) * "_R"
        DATA_I = uppercase(_correlation) * "_I"
        for row in 1:length(data_frame[:,DATA_R])
            index_num = data_frame[row,"INDEX"]
            o1 = index_frame[index_num,ID1]
            o2 = index_frame[index_num,ID2]
            τ = (n_τ != 1) ? index_frame[index_num,"Tau"]+1 : 1
            s1 = index_frame[index_num,dim_prefix*"1"]+1
            dimensions > 1 ? s2 = index_frame[index_num,dim_prefix*"2"]+1 : s2 = 1
            dimensions > 2 ? s3 = index_frame[index_num,dim_prefix*"3"]+1 : s3 = 1
            bin = data_frame[row,"BIN"]
            PID = data_frame[row,"BIN"] +1
            MEAN_R[o1,o2,τ,s1,s2,s3,bin,PID] = data_frame[row,DATA_R] 
            MEAN_I[o1,o2,τ,s1,s2,s3,bin,PID] = data_frame[row,DATA_I]
            SIGN_R[o1,o2,τ,s1,s2,s3,bin,PID] = data_frame[row,"SIGN_R"]
            SIGN_I[o1,o2,τ,s1,s2,s3,bin,PID] = data_frame[row,"SIGN_R"]  
        end
        return dimension_key, MEAN_R, MEAN_I, SIGN_R, SIGN_I 
    end
    println("If you read this, something has gone awry.")
end


function get_dimensions(;df::DataFrame,space::String)
    dimensions = 1
    data_names = names(df)
    if space == "momentum"
        if "K_3" in data_names
            dimensions = 3
        elseif "K_2" in data_names
            dimensions = 2
        end
    else
        if "R_3" in data_names
            dimensions = 3
        elseif "R_2" in data_names
            dimensions = 2
        end
    end
    return dimensions
end

