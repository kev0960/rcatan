use once_cell::sync::Lazy;

use crate::{Building, NodeId, RoadId, board::Board};

const TEMPLATE: &str = r#"
               51        52        53               
              /  \      /  \      /  \              
             r66  r67  r68  r69  r70  r71           
          47/      \48/      \49/      \50          
          |         |         |         |           
          r62       r63       r64       r65         
          |         |         |         |           
          43        44        45        46          
         /  \      /  \      /  \      /  \         
        r54  r55  r56  r57  r58  r59  r60  r61      
     38/      \39/      \40/      \41/      \42     
     |         |         |         |         |      
     r49       r50       r51       r52       r53    
     |         |         |         |         |      
     33        34        35        36        37     
    /  \      /  \      /  \      /  \      /  \    
   r39  r40  r41  r42  r43  r44  r45  r46  r47  r48 
27/      \28/      \29/      \30/      \31/      \32
|         |         |         |         |         | 
r33       r34       r35       r36       r37      r38
|         |         |         |         |         | 
21        22        23        24        25        26
  \      /  \      /  \      /  \      /  \      /  
   r23  r24  r25  r26  r27  r28  r29  r30  r31  r32 
    \16/      \17/      \18/      \19/      \20/    
     |         |         |         |         |      
     r18       r19       r20       r21       r22    
     |         |         |         |         |      
     11        12        13        14        15     
       \      /  \      /  \      /  \      /       
        r10  r11  r12  r13  r14  r15  r16  r17      
         \07/      \08/      \09/      \10/         
          |         |         |         |           
          r06       r07       r08       r09         
          |         |         |         |           
          03        04        05        06          
            \      /  \      /  \      /            
             r00  r01  r02  r03  r04  r05           
              \00/      \01/      \02/              "#;

static TEMPLATE_TOKENS: Lazy<Vec<TemplateToken>> =
    Lazy::new(|| parse_template());

enum TemplateToken {
    Literal(String),
    Road(RoadId),
    Structure(NodeId),
}

fn parse_template() -> Vec<TemplateToken> {
    let mut tokens = Vec::new();

    let mut current_literal = String::new();

    let mut chars = TEMPLATE.chars();
    while let Some(c) = chars.next() {
        if c == 'r' {
            if !current_literal.is_empty() {
                tokens.push(TemplateToken::Literal(current_literal));
                current_literal = String::new();
            }

            let digit_1 = chars.next().unwrap();
            let digit_2 = chars.next().unwrap();

            let road_id = RoadId::new(
                (digit_1.to_digit(10).unwrap() * 10
                    + digit_2.to_digit(10).unwrap()) as u8,
            );

            tokens.push(TemplateToken::Road(road_id));
        } else if c.is_digit(10) {
            if !current_literal.is_empty() {
                tokens.push(TemplateToken::Literal(current_literal));
                current_literal = String::new();
            }

            let digit_2 = chars.next().unwrap();

            let node_id = NodeId::new(
                (c.to_digit(10).unwrap() * 10 + digit_2.to_digit(10).unwrap())
                    as u8,
            );

            tokens.push(TemplateToken::Structure(node_id));
        } else {
            current_literal.push(c);
        }
    }

    if !current_literal.is_empty() {
        tokens.push(TemplateToken::Literal(current_literal));
    }

    tokens
}

pub fn render_board(board: &Board) -> String {
    let tokens = &*TEMPLATE_TOKENS;

    tokens
        .iter()
        .map(|t| match t {
            TemplateToken::Literal(s) => s.clone(),
            TemplateToken::Road(r) => {
                let road = board.get_road(*r);
                if let Some(player) = road.get() {
                    format!("{:<3}", usize::from(player))
                } else {
                    "   ".to_owned()
                }
            }
            TemplateToken::Structure(s) => {
                let structure = board.get_structure(*s);
                if let Some((building, player)) = structure.get() {
                    let building_char = match building {
                        Building::Settlement => 'S',
                        Building::City => 'C',
                    };

                    format!("{}{}", building_char, usize::from(player))
                } else {
                    "  ".to_owned()
                }
            }
        })
        .collect::<String>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PlayerId;

    #[test]
    fn template_parses_into_72_roads_and_54_structures() {
        let tokens = &*TEMPLATE_TOKENS;
        let road_count = tokens
            .iter()
            .filter(|t| matches!(t, TemplateToken::Road(_)))
            .count();
        let structure_count = tokens
            .iter()
            .filter(|t| matches!(t, TemplateToken::Structure(_)))
            .count();
        assert_eq!(road_count, 72);
        assert_eq!(structure_count, 54);
    }

    #[test]
    fn renders_empty_board_with_no_digits() {
        let board = Board::new(&[]);
        let output = render_board(&board);

        assert!(!output.is_empty());
        // With no roads/structures placed, every token slot is blank — the
        // rendered output should contain no digits at all.
        assert!(
            !output.chars().any(|c| c.is_ascii_digit()),
            "empty board should not contain any digits, got:\n{}",
            output
        );
    }

    #[test]
    fn renders_settlement() {
        let mut board = Board::new(&[]);
        board.build_structure(
            NodeId::new(0),
            PlayerId::new(0),
            Building::Settlement,
        );

        let output = render_board(&board);
        assert!(
            output.contains("S0"),
            "expected 'S0' for player 0 settlement, got:\n{}",
            output
        );
    }

    #[test]
    fn renders_city() {
        let mut board = Board::new(&[]);
        board.build_structure(
            NodeId::new(11),
            PlayerId::new(1),
            Building::City,
        );

        let output = render_board(&board);
        assert!(
            output.contains("C1"),
            "expected 'C1' for player 1 city, got:\n{}",
            output
        );
    }

    #[test]
    fn renders_road() {
        let mut board = Board::new(&[]);
        board.build_road(RoadId::new(0), PlayerId::new(2));

        let output = render_board(&board);
        // Roads render as `{:<3}` of the player id, so player 2 becomes "2  ".
        assert!(
            output.contains("2  "),
            "expected '2  ' for player 2 road, got:\n{}",
            output
        );
    }

    #[test]
    fn renders_fully_populated_board() {
        let mut board = Board::new(&[]);
        // Even nodes get settlements, odd nodes get cities. Player id cycles
        // through 0..=3 so each token slot is exercised.
        for n in 0..54u8 {
            let building = if n % 2 == 0 {
                Building::Settlement
            } else {
                Building::City
            };
            board.build_structure(
                NodeId::new(n),
                PlayerId::new(n % 4),
                building,
            );
        }
        for r in 0..72u8 {
            board.build_road(RoadId::new(r), PlayerId::new(r % 4));
        }

        let output = render_board(&board);

        // The template carries lots of trailing whitespace per line for visual
        // alignment, but editors / formatters tend to strip it from string
        // literals. Compare line-by-line with trailing whitespace stripped so
        // the test stays robust without losing meaningful content.
        let expected = r#"
               C3        S0        C1
              /  \      /  \      /  \
             2    3    0    1    2    3
          C3/      \S0/      \C1/      \S2
          |         |         |         |
          2         3         0         1
          |         |         |         |
          C3        S0        C1        S2
         /  \      /  \      /  \      /  \
        2    3    0    1    2    3    0    1
     S2/      \C3/      \S0/      \C1/      \S2
     |         |         |         |         |
     1         2         3         0         1
     |         |         |         |         |
     C1        S2        C3        S0        C1
    /  \      /  \      /  \      /  \      /  \
   3    0    1    2    3    0    1    2    3    0
C3/      \S0/      \C1/      \S2/      \C3/      \S0
|         |         |         |         |         |
1         2         3         0         1        2
|         |         |         |         |         |
C1        S2        C3        S0        C1        S2
  \      /  \      /  \      /  \      /  \      /
   3    0    1    2    3    0    1    2    3    0
    \S0/      \C1/      \S2/      \C3/      \S0/
     |         |         |         |         |
     2         3         0         1         2
     |         |         |         |         |
     C3        S0        C1        S2        C3
       \      /  \      /  \      /  \      /
        2    3    0    1    2    3    0    1
         \C3/      \S0/      \C1/      \S2/
          |         |         |         |
          2         3         0         1
          |         |         |         |
          C3        S0        C1        S2
            \      /  \      /  \      /
             0    1    2    3    0    1
              \S0/      \C1/      \S2/"#;

        let normalize = |s: &str| {
            s.lines()
                .map(str::trim_end)
                .collect::<Vec<_>>()
                .join("\n")
        };
        assert_eq!(normalize(&output), normalize(expected));
    }

    #[test]
    fn renders_multiple_buildings_for_different_players() {
        let mut board = Board::new(&[]);
        board.build_structure(
            NodeId::new(0),
            PlayerId::new(0),
            Building::Settlement,
        );
        board.build_structure(
            NodeId::new(20),
            PlayerId::new(1),
            Building::City,
        );
        board.build_road(RoadId::new(5), PlayerId::new(2));

        let output = render_board(&board);
        assert!(output.contains("S0"), "missing S0 in:\n{}", output);
        assert!(output.contains("C1"), "missing C1 in:\n{}", output);
        assert!(output.contains("2  "), "missing road '2  ' in:\n{}", output);

        println!("{}", output);
    }
}
