
Jmol._isAsync = false;
var jmolApplet0;

var Info = {
    width: 450,
    height: 450,
    debug: false,
    color: "0xFFFFFF",
    addSelectionOptions: false,
    use: "HTML5",
    // JAVA HTML5 WEBGL are all options
    j2sPath: "/static/jsmol/j2s",
    // XXX how coded for now.
    //serverURL: "http://chemapps.stolaf.edu/jmol/jsmol/php/jsmol.php",
    readyFunction: jmol_isReady,
    disableJ2SLoadMonitor: true,
    disableInitialConsole: true,
    allowJavaScript: false
};

function repeatCell(repeat)
{
    if (repeat == 221)
    {
        Jmol.script(jmolApplet0, 'load "" {2 2 1}');
    }
    else if (repeat == 222)
    {
        Jmol.script(jmolApplet0, 'load "" {2 2 2}');
    }
    else
    {
        Jmol.script(jmolApplet0, 'load "" {1 1 1}');
    }
}

$(document).ready(function()
{
    $("#appdiv").html(Jmol.getAppletHtml("jmolApplet0", Info))
})