library(tidyverse)

# Read metrics data
metrics <- read_csv("data/output/validation_metrics.csv") %>% 
    filter(Method != "Trend surface") %>% 
    mutate(
        Month = as.factor(Month),
        Region = if_else(Region == "b1", "Corn Belt", "Cropland"),
    ) %>% 
    select(-c(N, BIAS)) %>% 
    pivot_longer(c(RASPE, INT, DSS, MDSS), names_to = "Metric", values_to = "Value")

metrics_kriging  <- metrics %>% filter(Method == "Kriging")
metrics_cokriging <- metrics %>% filter(Method == "Cokriging")

value_diff <- metrics_kriging$Value - metrics_cokriging$Value
metrics_new <- metrics_cokriging %>% 
    mutate(Value = value_diff) %>% 
    mutate(Method = if_else(Value > 0, "Cokriging", "Kriging"))

# Plot metrics as bars with facets on region and metric
metrics_plot <- metrics_new %>% 
    ggplot(aes(x = Value, y = Month, fill = Method)) +
    geom_col() +
    geom_vline(xintercept = 0, linetype = "dashed", color = "grey60") +
    facet_grid(vars(Region), vars(fct_relevel(Metric, "RASPE", "INT", "DSS", "MDSS")),
        scales = "free_x"
    ) +
    scale_fill_manual(values = c("#5954d6", "#c0affb")) +
    scale_x_continuous(n.breaks = 4, expand = expansion(mult=0.08)) +
    scale_y_discrete(
        labels = c("202102" = "Feb", "202104" = "Apr", "202107" = "Jul", "202110" = "Oct")
    ) +
    labs(x = element_blank(), y = element_blank(), fill = "Favored Method") +
    theme_minimal() +
    theme(
        text=element_text(family="Montserrat", size=9),
        panel.grid.minor=element_blank(),
        legend.position="bottom",
        legend.spacing=unit(0, "lines"),
        legend.margin=margin(0, 0, 0, 0),
    )
metrics_plot
ggsave("figures/validation_metrics.png", width=16, height=7, units="cm")
