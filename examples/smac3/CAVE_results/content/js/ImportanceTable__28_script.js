
  (function() {
    var fn = function() {
      Bokeh.safely(function() {
        (function(root) {
          function embed_document(root) {
            
          var docs_json = '{"59cebe1b-a212-4007-89aa-857c0f90f492":{"roots":{"references":[{"attributes":{},"id":"1206","type":"StringEditor"},{"attributes":{"callback":null,"data":{"Ablation":["-0.00","96.30","45.21","-0.00","14.03"],"Forward-Selection":["00.33","-","-","00.31","-"],"LPI":["87.35 +/- 21.77","00.00 +/- 0.01","07.32 +/- 16.12","00.00 +/- 0.00","00.01 +/- 0.03"],"Parameters":["sp-var-dec-heur","sp-learned-clause-sort-heur","sp-orig-clause-sort-heur","sp-rand-var-dec-scaling","sp-restart-inc"],"fANOVA":["69.74 +/- 23.24","04.38 +/- 7.87","04.69 +/- 7.81","00.08 +/- 0.20","00.02 +/- 0.04"]},"selected":{"id":"1199","type":"Selection"},"selection_policy":{"id":"1198","type":"UnionRenderers"}},"id":"1190","type":"ColumnDataSource"},{"attributes":{},"id":"1207","type":"StringFormatter"},{"attributes":{"default_sort":"descending","editor":{"id":"1208","type":"StringEditor"},"field":"LPI","formatter":{"id":"1209","type":"StringFormatter"},"title":"LPI","width":100},"id":"1195","type":"TableColumn"},{"attributes":{"default_sort":"descending","editor":{"id":"1206","type":"StringEditor"},"field":"Forward-Selection","formatter":{"id":"1207","type":"StringFormatter"},"title":"Forward-Selection","width":100},"id":"1194","type":"TableColumn"},{"attributes":{},"id":"1205","type":"StringFormatter"},{"attributes":{},"id":"1203","type":"StringFormatter"},{"attributes":{"default_sort":"descending","editor":{"id":"1204","type":"StringEditor"},"field":"Ablation","formatter":{"id":"1205","type":"StringFormatter"},"title":"Ablation","width":100},"id":"1193","type":"TableColumn"},{"attributes":{},"id":"1198","type":"UnionRenderers"},{"attributes":{"default_sort":"descending","editor":{"id":"1202","type":"StringEditor"},"field":"fANOVA","formatter":{"id":"1203","type":"StringFormatter"},"title":"fANOVA","width":100},"id":"1192","type":"TableColumn"},{"attributes":{},"id":"1200","type":"StringEditor"},{"attributes":{},"id":"1209","type":"StringFormatter"},{"attributes":{},"id":"1208","type":"StringEditor"},{"attributes":{"columns":[{"id":"1191","type":"TableColumn"},{"id":"1192","type":"TableColumn"},{"id":"1193","type":"TableColumn"},{"id":"1194","type":"TableColumn"},{"id":"1195","type":"TableColumn"}],"height":170,"index_position":null,"source":{"id":"1190","type":"ColumnDataSource"},"view":{"id":"1197","type":"CDSView"}},"id":"1196","type":"DataTable"},{"attributes":{},"id":"1204","type":"StringEditor"},{"attributes":{},"id":"1202","type":"StringEditor"},{"attributes":{},"id":"1199","type":"Selection"},{"attributes":{"source":{"id":"1190","type":"ColumnDataSource"}},"id":"1197","type":"CDSView"},{"attributes":{},"id":"1201","type":"StringFormatter"},{"attributes":{"default_sort":"descending","editor":{"id":"1200","type":"StringEditor"},"field":"Parameters","formatter":{"id":"1201","type":"StringFormatter"},"sortable":false,"title":"Parameters","width":150},"id":"1191","type":"TableColumn"}],"root_ids":["1196"]},"title":"Bokeh Application","version":"1.1.0"}}';
          var render_items = [{"docid":"59cebe1b-a212-4007-89aa-857c0f90f492","roots":{"1196":"257e7510-f840-4075-ba69-f4847413a4bd"}}];
          root.Bokeh.embed.embed_items(docs_json, render_items);
        
          }
          if (root.Bokeh !== undefined) {
            embed_document(root);
          } else {
            var attempts = 0;
            var timer = setInterval(function(root) {
              if (root.Bokeh !== undefined) {
                embed_document(root);
                clearInterval(timer);
              }
              attempts++;
              if (attempts > 100) {
                console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                clearInterval(timer);
              }
            }, 10, root)
          }
        })(window);
      });
    };
    if (document.readyState != "loading") fn();
    else document.addEventListener("DOMContentLoaded", fn);
  })();
