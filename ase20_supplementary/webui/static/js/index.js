
let options = {
  name: 'breadthfirst',

  fit: true, // whether to fit the viewport to the graph
  directed: true, // whether the tree is directed downwards (or edges can point in any direction if false)
  padding: 30, // padding on fit
  circle: false, // put depths in concentric circles if true, put depths top down if false
  grid: true, // whether to create an even grid into which the DAG is placed (circle:false only)
  spacingFactor: 1.75, // positive spacing factor, larger => more space between nodes (N.B. n/a if causes overlap)
  boundingBox: undefined, // constrain layout bounds; { x1, y1, x2, y2 } or { x1, y1, w, h }
  avoidOverlap: true, // prevents node overlap, may overflow boundingBox if not enough space
  nodeDimensionsIncludeLabels: false, // Excludes the label when calculating node bounding boxes for the layout algorithm
  roots: undefined, // the roots of the trees
  maximal: false, // whether to shift nodes down their natural BFS depths in order to avoid upwards edges (DAGS only)
  animate: false, // whether to transition the node positions
  animationDuration: 500, // duration of animation in ms if enabled
  animationEasing: undefined, // easing of animation if enabled,
  animateFilter: function ( node, i ){ return true; }, // a function that determines whether the node should be animated.  All nodes animated by default on animate enabled.  Non-animated nodes are positioned immediately when the layout starts
  ready: undefined, // callback on layoutready
  stop: undefined, // callback on layoutstop
  transform: function (node, position ){ return position; } // transform a given node position. Useful for changing flow direction in discrete layouts
};

var cyto;
var select_options;
var attention_buffer;

var default_ast_html = '<div class=\"ui icon header\"><i class=\"list alternate outline icon\"></i>No statement is selected.</div>';
var last_ast_sel;
var last_att_sel;


function predict(){
  reset();

  button = $("#run");

  if(button.hasClass("primary")){
    failMsg("You have to wait till request is finished.");
    return;
  }

  var text = editor.getValue();

  $.ajax({
    url: '/api/task/',
    type: 'PUT',
    data: {data: text},
    success: function(data){
      startRequest(data);
    }
  }).fail(function(){
    failMsg("Something went wrong while connecting.");
  });

}

function handleUpdate(id, data){
  reset();
  if('exception' in data){
    failMsg(data['exception']);
  }else{

    if(data['pred']){
      loadPrediction(id);
      updateStatusMsg("Prediction finished. Start rendering.");
    }

    if(data['attention']){
      loadAttention(id);
      updateStatusMsg("Attention rendered.");
    }
  }
  if(data['finish']){
    loadFunctions(id);
  }
}

function statusUpdates(id){
  var finished = false;
  $.ajax({
    url: 'api/task/'+id+"/",
    type: 'GET',
    success: function(data){
      handleUpdate(id, data);
      finished = data['finish'];
    },
    complete: function(){
      if(!finished){
        setTimeout(function(){statusUpdates(id);}, 500);
      }
    }
  })

}

function updateStatusMsg(msg){
  dim = $('#result .dimmer');

  if(!dim.hasClass("active")){
    return;
  }

  dim.empty();
  dim.html("<div class=\"ui text loader\">"+msg+"</div>");
}

function startRequest(data){
  id = data['request_id'];
  statusUpdates(id);
}

function failMsg(msg){
  fc = $("#fail");
  $("#fail .header").text(msg);
  fc.removeClass('hidden');
  failedMode();
}

function reset(){
  fc = $("#fail");
  if(!fc.hasClass("hidden")){
    fc.addClass("hidden");
  }

}


function presentPrediction(id, data){
  $("#prediction").text(data[0]);
}

function loadPrediction(id){
  container = $('body');
  if(container.attr('prediction') == id)
    return;
  container.attr('prediction', id);

  $.ajax({
    url:"/api/prediction/"+id+"/",
    type: 'GET',
    success: function(data){presentPrediction(id, data);}
  }).fail(
    function(){
      failMsg("Cannot load prediction...");
      container.removeAttr('graph');
    }
  )
}

function get_pos(txt){

  pos_str = txt.split(":");

  pos = pos_str[pos_str.length - 2];

  return parseInt(pos);
}

function get_col(txt){

  pos_str = txt.split(":");

  pos = pos_str[pos_str.length - 1];

  return parseInt(pos);
}

function nonEmptyStmts(func_obj){

  positions = [];

  for(pos in func_obj){
    if(pos == 'attention' || pos.includes("NOOP")){
      continue;
    }
    positions.push(pos);
  }

  return positions;
}

function attentionBound(name, func_obj){
  min_att = 1.0;
  max_att = 0.000001;

  for(var pos in func_obj){
    if(pos == 'attention'){
      continue;
    }
    att = func_obj[pos]['attention'];

    if(att < min_att){
      min_att = att;
    }
    if(att > max_att){
      max_att = att;
    }
  }

  attention_buffer['state_bound'][name] = {'max': max_att, 'min': min_att};
}


function mergeAtt(state_att, key, pos, col, obj){
    att = obj['attention'];
    ast_att = obj['ast'];

    if(key in state_att){

      old = state_att[key]['ast'];
      for(k in ast_att){
        old[k+":"+col] = ast_att[k];
      }
      state_att[key]['attention'] = att;

    } else {
      state_att[key] = {'attention': att, 'pos': pos, 'ast': ast_att};
    }
}


function processFunc(name, func_obj){

  att = func_obj['attention'];

  if($("#"+name)){
    $("#"+name).text(name + " - "+ att.toFixed(3));
  }

  count = 0;
  attentionBound(name, func_obj);

  positions = nonEmptyStmts(func_obj);

  state_att = {};
  att_offset = 0;

  for(var postIx in positions){
    posKey = positions[postIx];
    obj = func_obj[posKey];
    pos = get_pos(posKey);
    col = get_col(posKey);

    if(posKey.includes("ParmVarDecl")){
      pos = 0;
      att_offset -= 1;
    } else if (att_offset < 0){
      pos = +pos + +att_offset + 1;
    }

    key = "L"+pos;
    mergeAtt(state_att, key, pos, col, obj);
  }

  attention_buffer['state_att'][name] = state_att;
}


function bufferAttention(id, data){
  attention_buffer = {'func_att': {}, 'state_att': {}, 'state_bound': {}};

  for(var func_name in data){
    func_obj = data[func_name];
    attention_buffer['func_att'][func_name] = func_obj['attention'];
    processFunc(func_name, func_obj);
  }


}

function loadAttention(id){
  container = $('body');
  if(container.attr('attention') == id)
    return;
  container.attr('attention', id);

  $.ajax({
    url:"/api/attention/"+id+"/",
    type: 'GET',
    success: function(data){bufferAttention(id, data);}
  }).fail(
    function(){
      failMsg("Cannot load attention...");
      container.removeAttr('attention');
    }
  )
}

function escapeHtml(unsafe) {
    return unsafe
         .replace(/&/g, "&amp;")
         .replace(/</g, "&lt;")
         .replace(/>/g, "&gt;")
         .replace(/"/g, "&quot;")
         .replace(/'/g, "&#039;");
 }

 function displayAST(func_name, ast_id, pos_id){
    ast_att = attention_buffer['state_att'][func_name][ast_id]['ast'];

    if(last_ast_sel){
      displayAttentionLine(func_name, last_att_sel, last_ast_sel);
    }

    lines = $("#codeout ol").children();

    var pos = parseInt(pos_id.substring(1));

    line = lines[pos];

    $(line).css(
      "background-color", "#2185d0"
    );

    $(line).children().css("color", "white");
    last_att_sel = ast_id;
    last_ast_sel = pos_id;

    data = [];
    for(keyix in ast_att){
      name = keyix.split(":")[0];
      data.push({'name': name, 'value': ast_att[keyix]});
    }

    data.sort((x, y) => y.value - x.value);

    ast_field = d3.select("#ast");
    dim = ast_field.node().getBoundingClientRect();

    width = dim.width;
    height = 600;
    margin = {
      "top": 40,
      "left": 200,
      "right": 60,
      "bottom": 350
    };

    ast_field.html("");

    x = d3.scaleLinear()
            .domain([0, d3.max(data, d => d.value)])
            .range([margin.left, width - margin.right]);

    format = x.tickFormat(20);

    y = d3.scaleBand()
          .domain(d3.range(data.length))
          .range([margin.top, height - margin.bottom])
          .padding(0.1);

    xAxis = g => g
      .attr("transform", `translate(0,${margin.top})`)
      .call(d3.axisTop(x).ticks(width / 80))
      .call(g => g.select(".domain").remove());

    yAxis = g => g
      .attr("transform", `translate(${margin.left},0)`)
      .call(d3.axisLeft(y).tickFormat(i => data[i].name).tickSizeOuter(0));

    svg = ast_field.append("svg").style("width", "100%").style("height", height+"px");

    svg.append("g")
        .attr("fill", "darkgreen")
      .selectAll("rect")
      .data(data)
      .join("rect")
        .attr("x", x(0))
        .attr("y", (d, i) => y(i))
        .attr("width", d => x(d.value) - x(0))
        .attr("height", y.bandwidth());

      svg.append("g")
        .attr("fill", "white")
        .attr("text-anchor", "end")
        .style("font", "12px sans-serif")
      .selectAll("text")
      .data(data)
      .join("text")
        .attr("x", d => x(d.value) - 4)
        .attr("y", (d, i) => y(i) + y.bandwidth() / 2)
        .attr("dy", "0.35em")
        .text(d => format(d.value));

      svg.append("g")
          .call(xAxis);

      svg.append("g")
          .call(yAxis);
 }

 function displayAttentionLine(name, att_id, posId){
   if(!attention_buffer){
     return;
   }

   bounds = attention_buffer['state_bound'][name];

   scaleColor = d3.scaleLinear()
                .domain([bounds.min, bounds.max])
                .range(["#9dc183", "#006600"]);
   scale = d3.scaleLinear()
               .domain([bounds.min, bounds.max])
               .range([0, 1]);

   states = attention_buffer['state_att'][name];

   lines = $("#codeout ol").children();

   var pos = parseInt(posId.substring(1));

   line = lines[pos];

   att = states[att_id]['attention'];

   $(line).css(
     "background-color", scaleColor(att)
   );

   if(scale(att) > 0.5){
     $(line).children().css("color", "white");
   } else {
     $(line).children().css("color", "");
   }
 }

 function hasArgs(line){
   children = $(line).children();
   openBrack = false;
   for (var i = 0; i < children.length; i++) {
    var currentChild = children.eq(i);
    type = currentChild.attr("class");

    if(!openBrack && type == 'pun'){
      openBrack = true;
    }else if(openBrack && type == 'typ'){
      return true;
    }

   }
   return false;
 }

 function isLineFilled(line){
    return $(line).children('.pln').text().trim().length > 3;
 }

 function isFilled(pos, line){
   if(pos == 0){
     return hasArgs(line);
   }
   nType = $(line).children('.pun').length;
   nType += $(line).children('.pln').length;

   if(nType == 2){
     return isLineFilled(line);
   }

   nAllType = $(line).children().length;

   return nAllType > 4 || nAllType - nType > 0;
 }


 function displayAttention(name){

   if(!attention_buffer){
     return;
   }

   if(!(name in attention_buffer['state_bound'])){
     return;
   }

   bounds = attention_buffer['state_bound'][name];

   scaleColor = d3.scaleLinear()
                .domain([bounds.min, bounds.max])
                .range(["#9dc183", "#006600"]);
   scale = d3.scaleLinear()
               .domain([bounds.min, bounds.max])
               .range([0, 1]);

   states = attention_buffer['state_att'][name];

   lines = $("#codeout ol").children();
   count = 0;

   for(pos = 0; pos < lines.length; pos++){

     line = lines[pos];

     if(!isFilled(pos, line)){
       continue;
     }
     pId = "L"+count;
     rId = "L"+pos;
     count = count + 1;

     if(!(pId in states)){
       continue;
     }

     att = states[pId]['attention'];

     $(line).attr(
       "onclick", "displayAST(\""+name+"\", \""+pId+"\",\""+rId+"\" )"
     );


     $(line).css(
       "background-color", scaleColor(att)
     );

     if(scale(att) > 0.5){
       $(line).children().css("color", "white");
     }

   }

 }


function displayFunctionBody(id, name, body){

  $("#funcselect").children().removeClass("active");
  $("#funcselect #"+name).addClass("active");

  normalMode();

  code = $("#codeout");
  code.empty();

  code.html(
    PR.prettyPrintOne(escapeHtml(body), "c", true)
  );
  displayAttention(name);
//  renderColorLegend(name);
}


function renderColorLegend(name){
  element = d3.select("#color-legend");
  element.html("");

  svg = element.append("svg");

  bounds = attention_buffer['state_bound'][name];

  scaleColor = d3.scaleLinear()
               .domain([bounds.min, bounds.max])
               .range(["#b3ffb3", "#006600"]);

  colorLegend = d3.legend.color()
                  .labelFormat(d3.formate(".0f"))
                  .scale(scaleColor)
                  .shapeWidth(100)
                  .shapeHeight(10);
  svg.append("g").call(colorLegend);

}

function showFunction(name){
  container = $('body');
  id = container.attr('functions');
  loadFunctionBody(id, name);
}


function displayFunction(id, data){

  funcs = data['functions'];

  menu = $("#funcselect");
  menu.children('.item').remove();

  var htmlText = "";

  att = {};
  if(attention_buffer){
    att = attention_buffer['func_att'];
  }

  for(name in funcs){
    var fName = funcs[name];
    postfix = "";

    if(fName in att){
      postfix = " - "+att[fName].toFixed(3);
    }

    htmlText += "<a class=\"item\" id=\""+fName+"\" onclick=\"showFunction(\'"+fName+"\');\">"+fName+postfix+"</a>\n";
  }

  menu.html(menu.html() + htmlText);

  loadFunctionBody(id, "main");

  normalMode();
}

function loadFunctionBody(id, name){

  processMode();
  updateStatusMsg("Load function "+name+"..");

  name = encodeURI(name);

  $.ajax({
    url:"/api/cfile/"+id+"/"+name+"/",
    type: 'GET',
    success: function(data){displayFunctionBody(id, name, data);}
  }).fail(
    function(){
      failMsg("Cannot load functions...");
      container.removeAttr('functions');
    }
  )

}


function loadFunctions(id){
  container = $('body');
  if(container.attr('functions') == id)
    return;
  container.attr('functions', id);

  $.ajax({
    url:"/api/cfile/"+id+"/",
    type: 'GET',
    success: function(data){displayFunction(id, data);}
  }).fail(
    function(){
      failMsg("Cannot load functions...");
      container.removeAttr('functions');
    }
  )

}

function processMode(){
  $("#run").removeClass("primary").addClass("negative")
            .attr("onclick", "stopProcess();");
  $("#run i").removeClass("play").addClass("x");
  $("#result .dimmer").addClass("active");
  updateStatusMsg("Processing code input");
}

function normalMode(){

  $("#run").removeClass("negative").addClass("primary")
            .attr("onclick", "startProcess();");
  $("#run i").addClass("play").removeClass("x");
  $("#result .dimmer").removeClass("active");
  $("#ast").html(default_ast_html);
  last_ast_sel = null;

}

function failedMode(){
  $("#run").removeClass("negative").addClass("primary")
            .attr("onclick", "startProcess();");
  $("#run i").addClass("play").removeClass("x");
  dim = $("#result .dimmer");
  dim.empty();
  dim.html("<h2 class=\"ui inverted header\">Failed prediction. Try again.</h2>");
}


function startProcess(){

  if(!$("#run").hasClass("primary")){
    return;
  }

  processMode();
  predict();
}

function stopProcess(){

  if(!$("#run").hasClass("negative")){
    return;
  }

  failedMode();

}


function initSite(){

}
