# This file plots viral loads from existing collected data files

library(ggplot2)
library(dplyr)
library(readxl)
library(tidyr)
library(patchwork)

wd <- getwd()

data_file_path <- file.path(wd,"Data-Master")

df_ViralLoad_1 <-
    read_excel(
        file.path(data_file_path, "Viral Load.xlsx"),
        sheet=2,
        range="B2:G98"
    )

df_ViralLoad_2 <-
    read_excel(
        file.path(data_file_path, "Viral Load.xlsx"),
        sheet=3,
        range="B2:G44"
    )

df_ViralLoad_3 <-
    read_excel(
        file.path(data_file_path, "Viral Load.xlsx"),
        sheet=4,
        range="B2:G20"
    )

df_ViralLoad_4 <-
    read_excel(
        file.path(data_file_path, "Viral Load.xlsx"),
        sheet=5,
        range="B2:H58"
    )

df_ViralLoad_5 <-
    read_excel(
        file.path(data_file_path, "Viral Load.xlsx"),
        sheet=6,
        range="B2:K10"
    )
    

df_ViralLoad_merged <-
    bind_rows(
        df_ViralLoad_1,
        df_ViralLoad_2,
        df_ViralLoad_3
    )

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


p1 <-
    df_ViralLoad_1 |>
#    filter(`Reading Type` %in% "mean") |>
    pivot_wider(names_from = `Reading Type`,
                values_from = `log Viral Burden (TCID50/mL)`) |>
ggplot(aes(x = `Time (days)`|> as.factor()))+
geom_errorbar(aes(ymin = `std low`, ymax = `std high`,
                  group = `Viral Strain`),
              position = position_dodge(width = 0.3),
              linewidth = 0.5,
              width = 0.25
              )+
geom_line(aes(y = mean, color = `Viral Strain`,
              group = `Viral Strain`),
              position = position_dodge(width = 0.3),
              linetype = 2
          ) +
geom_point(aes(y = mean, fill = `Viral Strain`),
               shape = 21,
               position = position_dodge(width = 0.3),
               size = 3)+
ylab("log10 TCID50/mL") +
theme_classic()+
themer+
xlab("Days Post Infection")

p2 <-
    df_ViralLoad_2 |>
#    filter(`Reading Type` %in% "mean") |>
    pivot_wider(names_from = `Reading Type`,
                values_from = `Log Viral Load (copies/mg)`) |>
ggplot(aes(x = `Time (days)`|> as.factor()))+
geom_errorbar(aes(ymin = `std low`, ymax = `std high`,
                  group = `Viral Strain`),
              position = position_dodge(width = 0.3),
              linewidth = 0.5,
              width = 0.25
              )+
geom_line(aes(y = mean, color = `Viral Strain`,
              group = `Viral Strain`),
              position = position_dodge(width = 0.3),
              linetype = 2
          ) +
geom_point(aes(y = mean, fill = `Viral Strain`),
               shape = 21,
               position = position_dodge(width = 0.3),
               size = 3)+
ylab("log10 Copies/mg") +
theme_classic()+
themer+
xlab("Days Post Infection")

p3 <-
    df_ViralLoad_3 |>
#    filter(`Reading Type` %in% "mean") |>
    pivot_wider(names_from = `Reading Type`,
                values_from = `Viral Load (TCID50/g lung)`) |>
ggplot(aes(x = `Time (days)`|> as.factor()))+
geom_errorbar(aes(ymin = `std low`, ymax = `std high`,
                  group = `Mice Species`),
              position = position_dodge(width = 0.3),
              linewidth = 0.5,
              width = 0.25
              )+
geom_line(aes(y = mean, color = `Mice Species`,
              group = `Mice Species`),
              position = position_dodge(width = 0.3),
              linetype = 2
          ) +
geom_point(aes(y = mean, fill = `Mice Species`),
               shape = 21,
               position = position_dodge(width = 0.3),
               size = 3)+
ylab("TCID50/g lung")+
theme_classic()+
themer+
xlab("Days Post Infection")

p4 <-
    df_ViralLoad_4 |>
    pivot_longer(
        cols= c(`1`, `2`, `3`),
        names_to = "Sample_ID",
        values_to = "Viral Load"
    ) |>
    filter(! `Viral Strain` %in% "VN1203 H5N1 (LOW)") |>
    mutate(logviralload = log10(`Viral Load`+1)) |>
    group_by(`Viral Strain`, `Time (days)`) |>
    mutate(meanlogviralload = mean(logviralload),
           logviralstd = sd(logviralload)) |>
    ungroup() |>
    mutate(ymin = pmax(meanlogviralload - logviralstd, 0),
           ymax = meanlogviralload + logviralstd) |> 
    ggplot(aes(x = `Time (days)`|> as.factor(), y = meanlogviralload))+
    geom_errorbar(aes(
        ymax = ymax,
        ymin = ymin,
        group = `Viral Strain`),
        position = position_dodge(width = 0.3),
        linewidth = 0.5,
        width = 0.25
        )+
    geom_line(aes(y = meanlogviralload, color = `Viral Strain`,
                  group = `Viral Strain`),
              position = position_dodge(width = 0.3),
              linetype = 2
              ) +
    geom_point(aes(y = meanlogviralload, fill = `Viral Strain`),
               shape = 21,
               position = position_dodge(width = 0.3),
               size = 3)+
ylab("PFU/g")+
theme_classic()+
themer+
xlab("Days Post Infection")

p5 <-
    df_ViralLoad_5 |>
    ggplot(aes(x = `Time (Days)`|> as.factor(), y = `log Virus (TCID50/mL)`))+
    geom_errorbar(aes(
        ymax = `log Virus (TCID50/mL)` + logVirusstd,
        ymin = pmax(`log Virus (TCID50/mL)` - logVirusstd,0),
        group = Sex
        ),
        position = position_dodge(width = 0.3),
        linewidth = 0.5,
        width = 0.25
        )+
    geom_line(aes(y = `log Virus (TCID50/mL)`, color = Sex,
                  group = Sex,
                  ),
              position = position_dodge(width = 0.3),
              linetype = 2
              ) +
    geom_point(aes(y = `log Virus (TCID50/mL)`, fill = Sex),
               shape = 21,
               position = position_dodge(width = 0.3),
               size = 3,
               )+
ylab("log10 TCID50/mL")+
theme_classic()+
themer+
xlab("Days Post Infection")

p_viral_load_combined <-
    ((p5|p2|p3)/(p1|p4))+
    plot_annotation(tag_levels = "A")+
    plot_layout(guides = "collect")
