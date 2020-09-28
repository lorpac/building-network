# install.packages("shiny", dependencies = TRUE)
# install.packages("reticulate", dependencies = TRUE)

library("shiny")
library("leaflet")
library("leaflet.extras")
library("comprehenr")
library("markdown")

# The icons have been downloaded from "Font Awesome" under the Creative Commons licence (https://fontawesome.com/license)

imgs_folder = "www/.temp"

box_style = "background-color:#F5F5F5; padding: 1px 1px; text-align:center"

ui <- fluidPage(
  # titlePanel("Buildings network"),
  br(),
  absolutePanel(class = "panel panel-default", style = "opacity: 0.9; z-index: 10;", fixed=TRUE, right=0, left="auto", top=0, bottom="auto", width = 300, 
    
    h5(textOutput(outputId = "status"), style = "font-family: Lucida Console; color: #1b61e4; text-align: center")
  ),
  
  fluidRow(
  column(6,
           wellPanel(includeMarkdown("www/intro.md"))
  ),
  column(6, verticalLayout(
    wellPanel(
      textInput("name", "Job name", "BuildingsNetwork")
    ),
    wellPanel(
      verticalLayout(
        fluidRow(
          column(4, numericInput("lat", "Latitude", 45.745591)),
          column(4, numericInput("long", "Longitude", 4.871167)),
          column(
            4,
            fluidRow(column(6, checkboxInput("save", "Save results", TRUE), align="center"),
                     column(6, actionButton(
                       "button",
                       "Run",
                       style = "font-size : medium; color: #28a428",
                       icon = icon("play", lib = "font-awesome")
                     ), align="left"))
            )
          
        ),
        br(),
        leafletOutput("map")
      )
    ))
    
    
  )),

  
  fluidRow(column(
    4,
    # h3(textOutput(outputId = "coordinates")),
    
    div(
      style = box_style ,
      h4("Download buildings"),
      br(),
      imageOutput(outputId = "buildings")
    )
  ),
  column(
    4,
    div(
      style = box_style,
      h4("Merge buildings"),
      br(),
      imageOutput(outputId = "merged")
    )
  ),
  
  column(
    4,
    div(
      style = box_style,
      h4("Assign nodes"),
      br(),
      imageOutput(outputId = "nodes")
    )
  )),
  br(),
  fluidRow(column(
    4,
    div(
      style = box_style,
      h4("Assign edges"),
      br(),
      imageOutput(outputId = "edges")
    )
  ),
  column(
    4,
    div(
      style = box_style,
      h4("Create the Buildings Network"),
      br(),
      imageOutput(outputId = "net")
    )
  ),
  column(
    4,
    div(
      style = box_style,
      h4("Color Buildings"),
      br(),
      imageOutput(outputId = "colored_buildings")
    )
  )),
  br(),
  
  fluidRow(column(
    12,
    downloadButton("downloadData", label = "Download"), align = "center"
  )),
  
  hr(),
  p(
    "Having fun? Try our Jupyter notebook!",
    a(
      href = "https://github.com/lorpac/building-network",
      icon("github", lib = "font-awesome"),
      style = "color : initial"
    ),
    a(href = "https://github.com/lorpac/building-network", "lorpac/building-network")
    
  )
  
)

server <- function(input, output, session) {
  session$onSessionEnded(stopApp) # stop app when closing browser tab
  
  output$coordinates <-
    renderText({
      paste("Coordinates:",
            "(",
            paste(rv$lat, rv$lng, sep = ", "),
            ")")
    })
  
  
  observe(
  output$map <- renderLeaflet({
    leaflet(options = leafletOptions(minZoom = 0, maxZoom = 18)) %>%
      addProviderTiles(providers$CartoDB.Voyager,
                       options = providerTileOptions(noWrap = TRUE)) %>%
      # addMarkers(lng = rv$lng,
      #            lat = rv$lat) %>%
      
      # conversion km to lat and long (in Lyon) from http://www.csgnetwork.com/degreelenllavcalc.html
      addRectangles(
        lng1 = rv$lng - 1 / 77.817524427469,
        lat1 = rv$lat - 1 / 111.14631466252709,
        lng2 = rv$lng + 1 / 77.817524427469,
        lat2 = rv$lat + 1 / 111.14631466252709,
        fillOpacity = 0.05
      ) %>%
      addScaleBar() %>%
      addSearchOSM(options = searchOptions(autoCollapse = FALSE)) %>%
      setView(lng = rv$lng,
            lat = rv$lat,
            zoom = rv$zoom)
  })
  )
  
  observeEvent(input$lat, {
    rv$lat <- input$lat
  })
  observeEvent(input$long, {
    rv$lng <- input$long
  })
  
  
  observeEvent(input$map_center, {
    center = input$map_center
    rv$lat <- center$lat
    rv$lng <- center$lng
    rv$zoom <- input$map_zoom
    updateTextInput(session, "lat", value = rv$lat)
    updateTextInput(session, "long", value = rv$lng)
  })
  
  
  rv <- reactiveValues()
  rv$status_text = "Ready"
  rv$download_src = "www/placeholder.png"
  rv$merge_src = "www/placeholder.png"
  rv$nodes_src = "www/placeholder.png"
  rv$edges_src = "www/placeholder.png"
  rv$net_src = "www/placeholder.png"
  rv$buildings_color_src = "www/placeholder.png"
  rv$lat <- isolate({
    input$lat
  })
  rv$lng <- isolate({
    input$long
  })
  rv$zoom <- 13
  
  output$buildings = renderImage({
    list(src = rv$download_src,
         alt = 'Buildings',
         height = '100%')
  }, deleteFile = FALSE)
  
  output$merged = renderImage({
    list(src = rv$merge_src,
         alt = 'Merged buildings',
         height = '100%')
  }, deleteFile = FALSE)
  
  output$nodes = renderImage({
    list(src = rv$nodes_src,
         alt = 'Nodes',
         height = '100%')
  }, deleteFile = FALSE)
  
  output$edges = renderImage({
    list(src = rv$edges_src,
         alt = 'Edges',
         height = '100%')
  }, deleteFile = FALSE)
  
  output$net = renderImage({
    list(src = rv$net_src,
         alt = 'Network',
         height = '100%')
  }, deleteFile = FALSE)
  
  output$colored_buildings = renderImage({
    list(
      src = rv$buildings_color_src,
      alt = 'Colored Buildings',
      height = '100%'
    )
  }, deleteFile = FALSE)
  
  rv$status = reactiveFileReader(100, session, "status", readLines)
  output$status = renderText(rv$status_text)
  
  observeEvent(rv$status(), {
    status = strtoi(isolate({
      rv$status()
    }))
    
    
    
    if (status == 0) {
      rv$download_src = "www/placeholder.png"
      rv$merge_src = "www/placeholder.png"
      rv$nodes_src = "www/placeholder.png"
      rv$edges_src = "www/placeholder.png"
      rv$net_src = "www/placeholder.png"
      rv$buildings_color_src = "www/placeholder.png"
      rv$status_text = "Downloading buildings..."
      rv$date <- Sys.Date()
      rv$time <- format(Sys.time(), "%Hh%Mm%Ss")
      rv$folder_name <- paste0(input$name,
                               "-",
                               rv$date,
                               "-",
                               rv$time)
    }
    
    if (status > 0) {
      rv$download_src = paste0(imgs_folder, "/buildings.png")
      rv$status_text = "Merging buildings..."
    }
    if (status > 1) {
      rv$merge_src = paste0(imgs_folder, "/merged.png")
      rv$status_text = "Assigning nodes..."
    }
    if (status > 2) {
      rv$nodes_src = paste0(imgs_folder, "/nodes.png")
      rv$status_text = "Assigning edges..."
    }
    if (status > 3) {
      rv$edges_src = paste0(imgs_folder, "/edges.png")
      rv$status_text = "Creating network..."
    }
    if (status > 4) {
      rv$net_src = paste0(imgs_folder, "/net.png")
      rv$status_text = "Coloring buildings..."
    }
    if (status > 5) {
      rv$buildings_color_src = paste0(imgs_folder, "/buildings_color.png")
      rv$status_text = "Finished."
      rv$file_list <- list.files(imgs_folder)
      rv$file_paths <-
        to_vec(for (f in rv$file_list)
          file.path(imgs_folder, f))
      
      if (isolate(input$save)) {
        destination_folder = file.path("results", rv$folder_name)
        
        dir.create(destination_folder, recursive = TRUE)
        
        for (f in rv$file_paths) {
          file.copy(f, destination_folder)
        }
        
      }
      
      Sys.sleep(3)
      rv$status_text = "Ready"
      
    }
  }, ignoreInit = TRUE)
  
  
  observeEvent(input$button, {
    if (Sys.info()["sysname"] == "Windows")
    {
      args = c("/c",
               "./run.ps1",
               "-lat",
               isolate(rv$lat),
               "-lng",
               isolate(rv$lng))
      system2("powershell", args, wait = FALSE)
    }
    
    else {
      args = c("run.sh",
               isolate(rv$lat),
               isolate(rv$lng))
      
      system2("bash", args, wait = FALSE)
    }
    
    
  }
  , ignoreInit = TRUE)
  
  output$downloadData <- downloadHandler(filename <- function() {
    paste(rv$folder_name, "tar", sep = ".")
  },
  
  content <- function(file) {
    tar(file, imgs_folder)
  })
}

onStop(function() {
  unlink(imgs_folder, recursive = TRUE)
})

shinyApp(ui = ui, server = server)
