document.addEventListener("DOMContentLoaded", function(event) {
    let dev_version = document.getElementById("dev-version");
    let old_version = document.getElementById("old-version");

    let on_scikit_hep = window.location.href.indexOf("scikit-hep.org/pyhf") > -1;
    let on_readthedocs = window.location.href.indexOf("pyhf.readthedocs.io") > -1;

    if(dev_version && on_scikit_hep){
        // are we not on readthedocs?
        dev_version.classList.add("version-warning");
    }

    if(old_version && on_readthedocs){
        // is the readthedocs page not the latest version?
        console.log('making request');
        $.ajax({
            type: "GET",
            url: "https://pyhf.readthedocs.io/",
            success: function(data, textStatus, resp){
                console.log(resp.getAllResponseHeaders());
                const version = resp.getResponseHeader("x-rtd-version") || 'here';
                if(window.location.href.indexOf(version) === -1){
                    document.getElementById("latest-version-link").text = version;
                    old_version.classList.add("version-warning");
                }
            }
        });
    }
});
