// This is a little script to format and rename all the data necessary for training, testing, and validation.
use std::env;
use std::process::Command;
use std::vec::Vec;

fn main() {
    let args: Vec<String> = env::args().collect();

    let names = generate_names_array();

    if args.len() < 2 {
        return;
    }

    match args[1].as_str() {
        "check" => {
            eprintln!("Checking data...");
            check_data(&names);
        }
        "refactor" => {
            eprintln!("Sorting data...");
            refactor_data(&names);
        }
        "divide" => {
            eprintln!(
                "{}",
                format!(
                    "Dividing data up... -- {} for test, {} for train, {} for validation",
                    args[2].as_str(),
                    args[3].as_str(),
                    args[4].as_str()
                )
            );
            divide_data(
                &names,
                args[2].as_str().parse::<usize>().unwrap(),
                args[3].as_str().parse::<usize>().unwrap(),
                args[4].as_str().parse::<usize>().unwrap(),
            );
        }
        _ => return,
    }
}

fn divide_data(strings: &Vec<String>, test: usize, train: usize, validation: usize) {
    // First I make train/, test/, and validation/ directories, then put official pokemon art into test/{pokedex number}/0.png,
    // and then I put last 20% of data/{pokedex number} into validation/ and last 10% into test/
    let _removing_old = Command::new("rm")
        .arg("-rf")
        .arg("test")
        .arg("train")
        .arg("validation")
        .output()
        .expect("RM-ing failed.");
    let _making_dir = Command::new("mkdir")
        .arg("test")
        .arg("train")
        .arg("validation")
        .output()
        .expect("Something went wrong with making dir");
    let mut pdn = 1;
    while pdn <= 151 {
        let ls = String::from_utf8(
            Command::new("ls")
                .arg(format!("{}/{}/", "data", pdn))
                .output()
                .expect(format!("Problem ls-ing at {}", pdn).as_str())
                .stdout,
        )
        .unwrap();
        let _making_dirs = Command::new("mkdir")
            .arg(format!("test/{}", pdn))
            .arg(format!("train/{}", pdn))
            .arg(format!("validation/{}", pdn))
            .output()
            .expect("Problem making dirs");
        let mut split_up = ls.as_str().lines();
        let amt = ls.as_str().lines().count() / (test + train + validation);
        let mut i = 0;
        eprintln!(
            "{}",
            format!(
                "For {} validation = {} train = {} test = {}",
                strings[pdn - 1],
                amt * validation,
                amt * train,
                amt * test
            )
        );
        while i < validation * amt {
            let _mv = Command::new("cp")
                .arg(format!(
                    "{}/{}/{}",
                    "data",
                    pdn,
                    split_up.next_back().unwrap()
                ))
                .arg(format!("{}/{}/", "validation", pdn))
                .output()
                .expect("Move failed");
            i += 1;
        }
        i = 0;
        while i < test * amt {
            let _mv = Command::new("cp")
                .arg(format!(
                    "{}/{}/{}",
                    "data",
                    pdn,
                    &split_up.next_back().unwrap()
                ))
                .arg(format!("{}/{}/", "test", pdn))
                .output()
                .expect("Move failed");
            i += 1;
        }
        let mut next = split_up.next();
        let _mv = Command::new("cp")
            .arg(format!("{}/{}{}", "official_art", pdn, ".jpg"))
            .arg(format!("{}/{}/{}", "test", pdn, "0.jpg"))
            .output()
            .expect("Move failed");
        loop {
            match next {
                Some(_) => {
                    let _mv = Command::new("cp")
                        .arg(format!("{}/{}/{}", "data", pdn, next.unwrap()))
                        .arg(format!("{}/{}/", "train", pdn))
                        .output()
                        .expect("Move failed");
                }
                None => break,
            }
            next = split_up.next();
        }
        pdn += 1;
    }
}

fn refactor_data(strings: &Vec<String>) {
    // This function goes from unzipped packing from the Kaggle datasets to a data/ directory with all images numbered
    let mut i = 0;
    let size = strings.len();
    let _data_dir = Command::new("mkdir")
        .arg("data")
        .output()
        .expect("Problem occured with mkdir call.");
    while i < size {
        let path = format!("{}{}", "data/", i + 1);
        let name = &strings[i];
        let _mk_indiviual_dir = Command::new("mkdir")
            .arg(&path)
            .output()
            .expect("Problem occured with mkdir call.");
        let ls = String::from_utf8(
            Command::new("ls")
                .arg(name)
                .output()
                .expect("Problem with ls call.")
                .stdout,
        )
        .unwrap();
        let mut j = 0;
        let mut iter = ls.as_str().lines();
        let mut next = iter.next();
        loop {
            match next {
                Some(_) => {
                    let _move_call = Command::new("mv")
                        .arg(format!("{}/{}", &name, next.unwrap()))
                        .arg(format!("{}{}/{}{}", "data/", i + 1, j, ".jpg"))
                        .output()
                        .expect("Move failed");
                }
                None => break,
            }
            j += 1;
            next = iter.next();
        }
        i += 1;
    }
}

fn check_data(strings: &Vec<String>) {
    for string in strings {
        let ls = String::from_utf8(
            Command::new("ls")
                .arg(string)
                .output()
                .expect("Problem with ls call.")
                .stdout,
        )
        .unwrap();
        let mut split_up = ls.as_str().lines();
        let next = split_up.next();
        match next {
            Some(_) => continue,
            None => eprintln!("No folder found for {}", string),
        }
    }
}

fn generate_names_array() -> Vec<String> {
    let v = [
        "Bulbasaur",
        "Ivysaur",
        "Venusaur",
        "Charmander",
        "Charmeleon",
        "Charizard",
        "Squirtle",
        "Wartortle",
        "Blastoise",
        "Caterpie",
        "Metapod",
        "Butterfree",
        "Weedle",
        "Kakuna",
        "Beedrill",
        "Pidgey",
        "Pidgeotto",
        "Pidgeot",
        "Rattata",
        "Raticate",
        "Spearow",
        "Fearow",
        "Ekans",
        "Arbok",
        "Pikachu",
        "Raichu",
        "Sandshrew",
        "Sandslash",
        "NidoranF",
        "Nidorina",
        "Nidoqueen",
        "NidoranM",
        "Nidorino",
        "Nidoking",
        "Clefairy",
        "Clefable",
        "Vulpix",
        "Ninetales",
        "Jigglypuff",
        "Wigglytuff",
        "Zubat",
        "Golbat",
        "Oddish",
        "Gloom",
        "Vileplume",
        "Paras",
        "Parasect",
        "Venonat",
        "Venomoth",
        "Diglett",
        "Dugtrio",
        "Meowth",
        "Persian",
        "Psyduck",
        "Golduck",
        "Mankey",
        "Primeape",
        "Growlithe",
        "Arcanine",
        "Poliwag",
        "Poliwhirl",
        "Poliwrath",
        "Abra",
        "Kadabra",
        "Alakazam",
        "Machop",
        "Machoke",
        "Machamp",
        "Bellsprout",
        "Weepinbell",
        "Victreebel",
        "Tentacool",
        "Tentacruel",
        "Geodude",
        "Graveler",
        "Golem",
        "Ponyta",
        "Rapidash",
        "Slowpoke",
        "Slowbro",
        "Magnemite",
        "Magneton",
        "Farfetchd",
        "Doduo",
        "Dodrio",
        "Seel",
        "Dewgong",
        "Grimer",
        "Muk",
        "Shellder",
        "Cloyster",
        "Gastly",
        "Haunter",
        "Gengar",
        "Onix",
        "Drowzee",
        "Hypno",
        "Krabby",
        "Kingler",
        "Voltorb",
        "Electrode",
        "Exeggcute",
        "Exeggutor",
        "Cubone",
        "Marowak",
        "Hitmonlee",
        "Hitmonchan",
        "Lickitung",
        "Koffing",
        "Weezing",
        "Rhyhorn",
        "Rhydon",
        "Chansey",
        "Tangela",
        "Kangaskhan",
        "Horsea",
        "Seadra",
        "Goldeen",
        "Seaking",
        "Staryu",
        "Starmie",
        "MrMime",
        "Scyther",
        "Jynx",
        "Electabuzz",
        "Magmar",
        "Pinsir",
        "Tauros",
        "Magikarp",
        "Gyarados",
        "Lapras",
        "Ditto",
        "Eevee",
        "Vaporeon",
        "Jolteon",
        "Flareon",
        "Porygon",
        "Omanyte",
        "Omastar",
        "Kabuto",
        "Kabutops",
        "Aerodactyl",
        "Snorlax",
        "Articuno",
        "Zapdos",
        "Moltres",
        "Dratini",
        "Dragonair",
        "Dragonite",
        "Mewtwo",
        "Mew",
    ];
    let mut b = Vec::<String>::new();
    let mut i = 0;
    while i < 151 {
        b.push(v[i].to_string());
        i += 1;
    }
    return b;
}
