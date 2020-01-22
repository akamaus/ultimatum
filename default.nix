with import <nixpkgs> {};

let python_env = python3.withPackages (ps: with ps;
      [
      ipython
      jupyter
      matplotlib
      numpy
      tqdm]);

    link = "python-env";
    shellHook = ''
         # if [ -d ${link} ]
         # then
         #   echo "Remove old link: ${link}"
         #   rm ${link};
         # fi
         # echo Create symbolic link ${link} to ${python_env}
         # ln -s ${python_env} ${link}
         nix-store --add-root ${link} --indirect -r ${python_env}
    '';

in python_env.env.overrideAttrs (x: { shellHook = shellHook; })
