# This is code to plot states to be used to reproduce Liparulo et al.
library(ggplot2)
library(dplyr)
library(readxl)
library(tidyr)
library(patchwork)


wd <- getwd()

data_file_path <- file.path(wd,"Data-Master")

df_Lv_etal <-
    read_excel(
        file.path(data_file_path, "States.xlsx"),
        sheet=1,
        range="B2:H380"
    )

df_Robinson_etal <-
    read_excel(
        file.path(data_file_path, "States.xlsx"),
        sheet=2,
        range="B2:AI10"
    )

df_Shoemaker_etal <-
    read_excel(
        file.path(data_file_path, "States.xlsx"),
        sheet=3,
    )

celltype_Lv_etal <-
    c("Total Cell Count",
      "Neutrophil Counts",
      "CD4 Counts",
      "CD8 Counts")

themer <-
    theme(panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.background = element_blank(),
                                        # axis.line = element_line,
                                        # text = element_text(size = 20, family = 'sans'),
          axis.title = element_text(size = 20),
          axis.text = element_text(size = 12, face = "bold"),
          plot.caption = element_text(size = 15),
          legend.text = element_text(size = 20),
          legend.title = element_text(size = 25, face = "bold"),
                                        # legend.background = element_rect(color = "black"),
          strip.text = element_text(size = 15, face = "bold" ),
                                        # strip.clip = "on",
          panel.border = element_rect(fill = NA)
          ) 

p1a <-
    df_Lv_etal |>
    select(-`State Raw`)|>
    pivot_wider(
        names_from = `Measurement Type`,
        values_from = `State Converted`
    ) |>
    filter(`State Name` %in% celltype_Lv_etal) |>
    ggplot(aes(x = as.factor(`Time (days)`)))+
    geom_errorbar(aes(ymin = `std low`,
                      ymax = `std high`,
                      group = `Viral Strain`),
                  position = position_dodge(width = 0.3),
                  linewidth = 0.5,
                  width = 0.25
                  )+
    geom_line(aes(y = mean,
                  color = `Viral Strain`,
                  group = `Viral Strain`),
              position = position_dodge(width = 0.3),
              linetype = 2
              )+
    geom_point(aes(y = mean,
                   fill = `Viral Strain`),
               shape = 21,
               position = position_dodge(width = 0.3),
               size = 3)+
    facet_wrap(~`State Name`,
               scales = "free")+
    ylab("Cell Counts")+
    xlab("Days Post Infection")+
    theme_classic()+
    themer

p1b <-
    df_Lv_etal |>
    select(-`State Raw`)|>
    pivot_wider(
        names_from = `Measurement Type`,
        values_from = `State Converted`
    ) |>
    filter(!`State Name` %in% celltype_Lv_etal) |>
    ggplot(aes(x = as.factor(`Time (days)`)))+
    geom_errorbar(aes(ymin = `std low`,
                      ymax = `std high`,
                      group = `Viral Strain`),
                  position = position_dodge(width = 0.3),
                  linewidth = 0.5,
                  width = 0.25
                  )+
    geom_line(aes(y = mean,
                  color = `Viral Strain`,
                  group = `Viral Strain`),
              position = position_dodge(width = 0.3),
              linetype = 2
              )+
    geom_point(aes(y = mean,
                   fill = `Viral Strain`),
               shape = 21,
               position = position_dodge(width = 0.3),
               size = 3)+
    facet_wrap(~`State Name`,
               scales = "free")+
    ylab("pg/mL")+
    xlab("Days Post Infection")+
    theme_classic()+
    themer

p1 <- (p1a + p1b)+
    plot_layout(guides = "collect",
                axes = "collect")

# finish p2 and p3 before the meeting starts
    
p2 <-
    bind_cols(
  df_Robinson_etal |>
    pivot_longer(
      cols = c(CCL2, `IL-6`),
      names_to = "State Names",
      values_to = "State Values"
    ),
  df_Robinson_etal |>
    select(`Time (days)`, CCL2stdev, `IL-6stdev`) |>
    pivot_longer(
      cols = c(CCL2stdev, `IL-6stdev`),
      names_to = "Std Names",
      values_to = "Std Values"
    ) |>
    mutate(
      `State Names` = ifelse(startsWith(`Std Names`, "CCL2"), "CCL2", "IL-6")
    ) |>
    select(`Std Values`)
  ) |> glimpse()

    
mutate(ymin = `State Values` - `Std Values`) |> 
View()
    ggplot(aes(x = `Time (days)`|> as.factor()))+
    geom_errorbar(aes(ymin = `State Values` - `Std Values`,
                      ymax = `State Values` + `Std Values`,
                  group = Sex),
              position = position_dodge(width = 0.3),
              linewidth = 0.5,
              width = 0.25
              )+
    geom_line(aes(y = `State Values`, color = Sex,
                  group = Sex),
              position = position_dodge(width = 0.3),
              linetype = 2
              ) +
    geom_point(aes(y = `State Values`, fill = Sex),
               shape = 21,
               position = position_dodge(width = 0.3),
               size = 3)+
ylab("log10 TCID50/mL") +
theme_classic()+
xlab("Days Post Infection")

    
