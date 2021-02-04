document.addEventListener("DOMContentLoaded", function(event) {
    // are we not on readthedocs?
    if(window.location.href.indexOf("pyhf.readthedocs.io") === -1){
        document.getElementById("dev-version").classList.add("version-warning");
    } else {
        // is the readthedocs page not the latest version?
        let resp = $.ajax({type: "GET", url: "https://pyhf.readthedocs.io/"});
        let version = resp.getResponseHeader("x-rtd-version");
        if(window.location.href.indexOf(version) === -1){
            document.getElementById("latest-version-link").text = version;
            document.getElementById("old-version").classList.add("version-warning");
        }
    }
});
