from dask.distributed import LocalCluster
import numpy as np
from numpy import dtype, nan
import xarray as xr

regions = ["AK", "AN", "GL"]
variables = [
    "snow_depth",
    "snowfall",
    "snowmelt"
]
short_names = [
    "sd",
    "sf",
    "smlt",
]
era5_encoding = {"dtype": dtype("float32"), "zlib": True, "_FillValue": np.float32(nan), "complevel": 1}


"""
Temporal aggregation for 0.25-degree:
(0.25, hour) -> (0.25, day) -> (0.25, month) -> (0.25, year)
"""
def time_driver():
    cluster = LocalCluster(n_workers=10)
    client = cluster.get_client()

    for region in regions:
        for variable, short_name in zip(variables, short_names):
            base_file_name = f"{variable}_{region}_2015-2024"
            print(f"Processing {base_file_name}")
            ds = xr.open_dataset(
                f"/data/{base_file_name}.nc", chunks={"time": 24 * 1, "latitude": 721, "longitude": 1440}
            )

            # 0.25, (day, month, year), mean
            ds_025_day_mean = ds.resample(valid_time="D").mean()
            ds_025_day_mean = ds_025_day_mean.compute()
            ds_025_day_mean.to_netcdf(f"{base_file_name}_025day_mean.nc", encoding={short_name: era5_encoding})
            ds_025_month_mean = ds_025_day_mean.resample(valid_time="ME").mean()
            ds_025_month_mean.to_netcdf(f"{base_file_name}_025month_mean.nc", encoding={short_name: era5_encoding})
            ds_025_year_mean = ds_025_month_mean.resample(valid_time="YE").mean()
            ds_025_year_mean.to_netcdf(f"{base_file_name}_025year_mean.nc", encoding={short_name: era5_encoding})

            # 0.25, (day, month, year), min
            ds_025_day_min = ds.resample(valid_time="D").min()
            ds_025_day_min = ds_025_day_min.compute()
            ds_025_day_min.to_netcdf(f"{base_file_name}_025day_min.nc", encoding={short_name: era5_encoding})
            ds_025_month_min = ds_025_day_min.resample(valid_time="ME").min()
            ds_025_month_min.to_netcdf(f"{base_file_name}_025month_min.nc", encoding={short_name: era5_encoding})
            ds_025_year_min = ds_025_month_min.resample(valid_time="YE").min()
            ds_025_year_min.to_netcdf(f"{base_file_name}_025year_min.nc", encoding={short_name: era5_encoding})

            # 0.25, (day, month, year), max
            ds_025_day_max = ds.resample(valid_time="D").max()
            ds_025_day_max = ds_025_day_max.compute()
            ds_025_day_max.to_netcdf(f"{base_file_name}_025day_max.nc", encoding={short_name: era5_encoding})
            ds_025_month_max = ds_025_day_max.resample(valid_time="ME").max()
            ds_025_month_max.to_netcdf(f"{base_file_name}_025month_max.nc", encoding={short_name: era5_encoding})
            ds_025_year_max = ds_025_month_max.resample(valid_time="YE").max()
            ds_025_year_max.to_netcdf(f"{base_file_name}_025year_max.nc", encoding={short_name: era5_encoding})

    client.close()
    cluster.close()


"""
Spatial aggregation for daily, monthly, and yearly data:
(0.25, day) -> (0.5, day)
(0.25, day) -> (1, day)
(0.25, month) -> (0.5, month)
(0.25, month) -> (1, month)
(0.25, year) -> (0.5, year)
(0.25, year) -> (1, year)
"""
def space_driver():
    for region in regions:
        for variable, short_name in zip(variables, short_names):
            base_file_name = f"{variable}_{region}_2015-2024"
            print(f"Processing {base_file_name}")
            for time in ["day", "month", "year"]:
                for space in ["05", "1"]:
                    if space == "05":
                        coarse = 2
                    elif space == "1":
                        coarse = 4
                    else:
                        raise ValueError("space must be 05 or 1")

                    # max
                    ds_time_max = xr.open_dataset(
                        f"{base_file_name}_025{time}_max.nc", chunks={"time": 1, "latitude": 721, "longitude": 1440}
                    )
                    ds_space_time_max = ds_time_max.coarsen(latitude=coarse, longitude=coarse, boundary="trim").max()
                    ds_space_time_max.to_netcdf(
                        f"{base_file_name}_{space}{time}_max.nc", encoding={short_name: era5_encoding}
                    )

                    # min
                    ds_time_min = xr.open_dataset(
                        f"{base_file_name}_025{time}_min.nc", chunks={"time": 1, "latitude": 721, "longitude": 1440}
                    )
                    ds_space_time_min = ds_time_min.coarsen(latitude=coarse, longitude=coarse, boundary="trim").min()
                    ds_space_time_min.to_netcdf(
                        f"{base_file_name}_{space}{time}_min.nc", encoding={short_name: era5_encoding}
                    )

                    # mean
                    ds_time_mean = xr.open_dataset(
                        f"{base_file_name}_025{time}_mean.nc", chunks={"time": 1, "latitude": 721, "longitude": 1440}
                    )
                    ds_space_time_mean = ds_time_mean.coarsen(latitude=coarse, longitude=coarse, boundary="trim").mean()
                    ds_space_time_mean.to_netcdf(
                        f"{base_file_name}_{space}{time}_mean.nc", encoding={short_name: era5_encoding}
                    )

"""
Spatial aggregation for hourly data: need to be separated from the above one, as it eats up too much memory
(0.25, hour) -> (0.5, hour)
(0.25, hour) -> (1, hour)
"""
def space2_driver():
    cluster = LocalCluster(n_workers=10)
    client = cluster.get_client()

    for region in regions:
        for variable, short_name in zip(variables, short_names):
            base_file_name = f"{variable}_{region}_2015-2024"
            print(f"Processing {base_file_name}")
            ds = xr.open_dataset(f"/data/{base_file_name}.nc", chunks={"time": 1, "latitude": 721, "longitude": 1440})
            # 0.5, hour, mean
            ds_05_hour_mean = ds.coarsen(latitude=2, longitude=2, boundary="trim").mean()
            ds_05_hour_mean = ds_05_hour_mean.compute()
            ds_05_hour_mean.to_netcdf(f"{base_file_name}_05hour_mean.nc", encoding={short_name: era5_encoding})

            # 0.5, hour, max
            ds_05_hour_max = ds.coarsen(latitude=2, longitude=2, boundary="trim").max()
            ds_05_hour_max = ds_05_hour_max.compute()
            ds_05_hour_max.to_netcdf(f"{base_file_name}_05hour_max.nc", encoding={short_name: era5_encoding})

            # 0.5, hour, min
            ds_05_hour_min = ds.coarsen(latitude=2, longitude=2, boundary="trim").min()
            ds_05_hour_min = ds_05_hour_min.compute()
            ds_05_hour_min.to_netcdf(f"{base_file_name}_05hour_min.nc", encoding={short_name: era5_encoding})

            # 1, hour, mean
            ds_1_hour_mean = ds.coarsen(latitude=4, longitude=4, boundary="trim").mean()
            ds_1_hour_mean = ds_1_hour_mean.compute()
            ds_1_hour_mean.to_netcdf(f"{base_file_name}_1hour_mean.nc", encoding={short_name: era5_encoding})

            # 1, hour, max
            ds_1_hour_max = ds.coarsen(latitude=4, longitude=4, boundary="trim").max()
            ds_1_hour_max = ds_1_hour_max.compute()
            ds_1_hour_max.to_netcdf(f"{base_file_name}_1hour_max.nc", encoding={short_name: era5_encoding})

            # 1, hour, min
            ds_1_hour_min = ds.coarsen(latitude=4, longitude=4, boundary="trim").min()
            ds_1_hour_min = ds_1_hour_min.compute()
            ds_1_hour_min.to_netcdf(f"{base_file_name}_1hour_min.nc", encoding={short_name: era5_encoding})

    client.close()
    cluster.close()


if __name__ == "__main__":
    time_driver()
    space_driver()
    space2_driver()
