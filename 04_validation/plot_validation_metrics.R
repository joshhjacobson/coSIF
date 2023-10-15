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

# Make plots for each metric
metrics_plot <- metrics %>% 
    ggplot(aes(x = Value, y = Month, fill = Method)) +
    geom_col(position = position_dodge2(reverse = TRUE, padding = 0)) +
    facet_grid(vars(Region), vars(fct_relevel(Metric, "RASPE", "INT", "DSS", "MDSS")),
        scales = "free_x"
    ) +
    scale_fill_manual(values = c("#238b45", "#bae4b3")) +
    scale_y_discrete(
        labels = c("202102" = "Feb", "202104" = "Apr", "202107" = "Jul", "202110" = "Oct")
    ) +
    labs(x = element_blank(), y = element_blank()) +
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
