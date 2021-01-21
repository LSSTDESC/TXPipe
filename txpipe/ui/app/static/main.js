var pipeline_graph;
var render = new dagreD3.render();
var socket;

$( function() {
    $( "#tabs" ).tabs({activate: select_tab});
    $( "#accordion" ).accordion({activate: select_accordion});

    $("#select_pipeline_input").change(select_pipeline);
    socket = io();
    socket.on('log_javascript', log_javascript);
    socket.on('parsed_pipeline', parsed_pipeline);
    socket.on('no_pipeline', no_pipeline);
    socket.on('stage_launched', stage_launched);
    socket.on('stage_complete', stage_complete);
    socket.on('stage_failed', stage_failed);
    socket.on('stage_failed', stage_failed);
    socket.on('read_log', read_log);

} );

function select_tab(event, ui){
    console.log("tab");
    console.log(ui);
}

function select_accordion(event, ui){
    var stage = ui.newHeader[0].innerText;
    socket.emit("request_log", stage);

}

function get_text_width(text, font) {
    // re-use canvas object for better performance
    var canvas = get_text_width.canvas || (get_text_width.canvas = document.createElement("canvas"));
    var context = canvas.getContext("2d");
    context.font = font;
    var metrics = context.measureText(text);
    console.log(text, metrics.width);
    return Math.round(metrics.width * 1.4);
}

function visualize_graph(nodes, edges) { 
    // Create a new directed graph (global)
    pipeline_graph = new dagre.graphlib.Graph();

    // Set an object for the graph label
    pipeline_graph.setGraph({nodesep: 5, ranksep: 30});
    pipeline_graph.setDefaultEdgeLabel(function() { return {}; });

    // Default to assigning a new object as a label for each new edge.
    pipeline_graph.setDefaultEdgeLabel(function() { return {}; });

    // Create each node
    nodes.forEach(function(node) {
        console.log(node);
        pipeline_graph.setNode(node, { label: node,  width: get_text_width(node, 14), height: 5 });
    });

    // Create each edge
    edges.forEach(function(edge) {
        pipeline_graph.setEdge(edge[0], edge[1], {curve: d3.curveBasis});
    });



    // Compute the layout
    dagre.layout(pipeline_graph)

    render_pipeline();

    // // Run the renderer. This is what draws the final graph.
    // render(d3.select("svg g"), pipeline_graph);

    // // // Center the graph
    // // Set up an SVG group so that we can translate the final graph.
    // // var svg = d3.select("svg");
    // // var xCenterOffset = (svg.attr("width") - g.graph().width) / 2;
    // // svgGroup.attr("transform", "translate(" + xCenterOffset + ", 20)");
    // // svg.attr("height", g.graph().height + 40);

    // console.log(d3.selectAll('svg g.comp'));



}


function append_listing(name, content){

    bars = '<h3><a href="#">' + name + '</a></h3><div><p ' + 'id="log-' + name + '">' + content + '</p></div>';
    console.log(bars);
    var acc = $("#accordion");
    acc.append(bars);

    // reset
    acc.accordion('destroy');
    acc.accordion({activate: select_accordion});

}

function read_log(d){
    console.log(d);
    $("#log-" + d['stage']).html("<pre>\n" + d['data'] + "\n</pre>")
}


function render_pipeline(){
    if (!pipeline_graph) return;
    render(d3.select("svg g"), pipeline_graph);
    d3.selectAll("svg g.node").on("click", click_node);

}

function launch_pipeline(evt){
    socket.emit("launch_pipeline", {}) 
}

function click_node(evt){
    var node = pipeline_graph.node(evt);
    console.log(node);
}

function no_pipeline(evt){
    alert("No pipeline loaded");
}


function log_javascript(evt){
    console.log(evt);
}

function log_python(msg){
    socket.emit("log_python", msg)
}

function stage_launched(d){
    set_node_colour(d['stage'], "#ffbf00");

    // Append a tab for the log file
    append_listing(d['stage'], '')
}

function stage_complete(d){
    set_node_colour(d['stage'], "#80ff00");
}

function stage_failed(d){
    set_node_colour(d['stage'], "#ff4000");
}

function set_node_colour(node, colour){
    pipeline_graph.setNode(
        node, {label: node,  width: get_text_width(node, 14), height: 5, style: "fill:"+colour});
    render_pipeline()

}

function select_pipeline(){
    log_python("Selected pipeline")
    var file = $("#select_pipeline_input").prop("files")[0];
    var reader = new FileReader();
    reader.onload = function(evt){
        socket.emit("loaded_pipeline", {"yml": evt.target.result});
    };
    reader.readAsText(file);
}

function loaded_pipeline(stages){
    stages.forEach(element => console.log(element["name"]));
}


function parsed_pipeline(data){
    console.log(data);
    visualize_graph(data['nodes'], data['edges']);
}