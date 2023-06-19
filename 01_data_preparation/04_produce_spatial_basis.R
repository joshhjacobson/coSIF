# Produce the fixed spatial basis using FRK with bisquare basis functions on the
# 0.05-degree CMG grid

library(tidyverse)
library(FRK)

library(sp)
library(sf)
library(rnaturalearth)
library(rnaturalearthdata)
theme_set(theme_bw())

setwd("~/coSIF/01_data_preparation")

## Define custom plotting functions
circleFun <- function(center = c(0,0),diameter = 1, npoints = 100){
  r = diameter / 2                            # radius
  tt <- seq(0,2*pi,length.out = npoints)      # default 100 points on circle
  xx <- center[1] + r * cos(tt)               # x values
  yy <- center[2] + r * sin(tt)               # y values
  return(data.frame(x = xx, y = yy))          # return in data frame
}

show_basis_custom <- function(basis, g){
  l <- lapply(1:basis@n,function(i) {   # for each basis function
    
    ## Create a data frame containing the x,y coordinates of a circle
    ## around the basis function centroid and the function's resolution
    data.frame(circleFun(center=as.numeric(basis@df[i,1:2]),
                          diameter = basis@df$scale[i]),
               res=basis@df$res[i],
               id = i)})
  df <- bind_rows(l)              # quick rbind of l
  df$res <- as.factor(df$res)     # convert to factor
  
  ## Draw circles with different linetypes for the different resolutions
  g <- g + geom_path(data=df,
                     aes(x=x,y=y,group=id,linetype="dashed"))
}


### Get gridded data for data locations
df_data_sif <- read_csv(
  "../data/intermediate/OCO2_005deg_months2021_north_america_SIF.csv", 
  col_select=c("lon", "lat", "sif")
)
df_data_xco2 <- read_csv(
  "../data/intermediate/OCO2_005deg_months2021_north_america_XCO2.csv", 
  col_select=c("lon", "lat", "xco2")
)

data_sp_sif <- df_data_sif
coordinates(data_sp_sif) <-  ~lon + lat

data_sp_xco2 <- df_data_xco2
coordinates(data_sp_xco2) <-  ~lon + lat


### Construct the basis functions
G_sif <- auto_basis(manifold = plane(),
                data = data_sp_sif,
                type = "bisquare",
                nres = 1,
                regular = 2,
                prune = 2,
                )

G_xco2 <- auto_basis(manifold = plane(),
                data = data_sp_xco2,
                type = "bisquare",
                nres = 1,
                regular = 2,
                prune = 2,
)

sif_xco2_common_idx <- seq(1, 60)[-c(1, 10, 20)]
G_xco2_only <- remove_basis(G_xco2, sif_xco2_common_idx)

# Plot the basis functions on a map
world <- ne_countries(scale = "medium", returnclass = "sf")
lakes <- ne_download(scale = "medium", type = "lakes", category = "physical", returnclass = "sf")

f0 <- ggplot() +
  geom_sf(data = world, fill = "antiquewhite") +
  geom_sf(data = lakes, fill = "aliceblue") +
  coord_sf(xlim = c(-135, -55), ylim = c(15, 65), 
           crs = sf::st_crs(4326), default_crs = sf::st_crs(4326)) +
  geom_rect(aes(xmin = -125, xmax = -65, ymin = 22, ymax = 58),
            fill = "transparent", color = "#EE2C2C", linewidth = 1) +
  xlab(NULL) + ylab(NULL)

f1 <- show_basis(G_sif, f0)
show_basis_custom(G_xco2_only, f1) +  theme(
  panel.grid.major = element_line(
    color = gray(.5),
    linetype = "dashed", 
    linewidth = 0.5
  ), 
  panel.background = element_rect(fill = "aliceblue"),
  legend.position = "none",
  plot.margin = margin(0.5, 0.5, 0.5, 0.5, "cm"),
)

ggsave("../figures/basis_functions.pdf", width = 6.80, height = 5.36)


### Evaluate the basis on the full grid

# Setup the grid
lon <- seq(-124.975, -65.025, 0.05)
lat <- seq(22.025, 57.975, 0.05)
grid <- expand_grid(lon=lon, lat=lat) %>% as.matrix()

# Evaluate the basis
X <- eval_basis(G_xco2, grid)
Xmat <- as.matrix(X)

dim(Xmat)

colnames(Xmat) <- sapply(paste(seq(dim(Xmat)[2])), function(x) paste("B", x, sep=""))
df_basis <- data.frame(cbind(grid, Xmat))

single_basis <- data.frame(lon = df_basis$lon,
                           lat = df_basis$lat,
                           z = df_basis$B16)

ggplot(sample_n(single_basis, 10000)) + geom_point(aes(x = lon, y = lat, col = z ))


### Save basis matrix to csv
write_csv(df_basis, "../data/intermediate/bisquares_basis.csv")
