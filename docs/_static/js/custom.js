document.addEventListener("DOMContentLoaded", function() {
    let dev_version = document.getElementById("dev-version");
    let on_scikit_hep = window.location.href.indexOf("scikit-hep.org/pyhf") > -1;

    if(dev_version && on_scikit_hep){
        // are we not on readthedocs?
        dev_version.classList.add("version-warning");
    }
});
